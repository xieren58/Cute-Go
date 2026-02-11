import { OnnxEngine, type AnalysisResult } from '../utils/onnx-engine';
import { MicroBoard, type Sign } from '../utils/micro-board';
import { 
    getCandidateMoves, 
    getGomokuScore, 
    checkGomokuWin, 
    evaluatePositionStrength, 
    GOMOKU_SCORES,
    attemptMove, // [New]
    getBoardHash // [New]
} from '../utils/goLogic';
import { BoardState, Player, Point } from '../types';


// Define message types
type WorkerMessage = 
    | { type: 'init'; payload: { 
        modelPath: string; 
        modelParts?: string[]; 
        wasmPath?: string; 
        numThreads?: number;
        onlyRules?: boolean; // [New]
    } }
    | { type: 'compute'; data: { 
            board: any[][]; // BoardState
            history: any[]; // HistoryItem[]
            color: 'black' | 'white';
            size: number;
            gameType?: 'Go' | 'Gomoku'; // [New]
            simulations?: number;
            komi?: number;
            difficulty?: 'Easy' | 'Medium' | 'Hard';
            temperature?: number;
      } }
    | { type: 'stop' }
    | { type: 'release' }
    | { type: 'reinit' };

let engine: OnnxEngine | null = null;
let initPromise: Promise<void> | null = null;
let initWatchdog: any = null;
const WATCHDOG_TIMEOUT = 30000; // 30s safety net

const clearWatchdog = () => {
    if (initWatchdog) {
        clearTimeout(initWatchdog);
        initWatchdog = null;
    }
};

const ctx: Worker = self as any;

// [Fix] Catch global script errors (e.g. Import failures)
ctx.onerror = (e) => {
    const msg = e instanceof ErrorEvent ? e.message : 'Unknown Worker Error';
    ctx.postMessage({ type: 'error', message: `脚本加载失败: ${msg}` });
};

// [Fix] Signal that worker script loaded successfully
ctx.postMessage({ type: 'status', message: 'Worker 线程已启动...' });

ctx.onmessage = async (e: MessageEvent<WorkerMessage>) => {
    const msg = e.data;

    try {
        if (msg.type === 'init') {
            const { modelPath, modelParts, wasmPath, numThreads, onlyRules } = msg.payload;
            
            // Cache config for Re-Init
            (self as any).aiConfig = msg.payload;

            // Dispose existing engine if any
            if (engine) engine.dispose();
            engine = null;

            if (onlyRules) {
                console.log("[AI Worker] Initialization Start (Rule-only Mode)");
                // Minor delay to ensure message order
                setTimeout(() => {
                    console.log("[AI Worker] Initialization Complete (Rule-only Mode)");
                    ctx.postMessage({ type: 'init-complete' });
                }, 50);
                return;
            }

            console.log("[AI Worker] Initializing OnnxEngine...");
            clearWatchdog();
            initWatchdog = setTimeout(() => {
                console.error("[AI Worker] Initialization Watchdog Triggered (Timeout)");
                ctx.postMessage({ type: 'error', message: 'Worker 初始化超时 (30s)' });
                initPromise = null;
            }, WATCHDOG_TIMEOUT);

            engine = new OnnxEngine({
                modelPath: modelPath,
                modelParts: modelParts, // Pass split parts
                wasmPath: wasmPath,
                numThreads: numThreads,
                debug: true // Enable debug for now
            });
            
            // [Lock] Prevent race conditions
            initPromise = engine.initialize((statusMsg) => {
                ctx.postMessage({ type: 'status', message: statusMsg });
            });
            
            await initPromise;
            initPromise = null; // Unlock
            clearWatchdog();

            console.log("[AI Worker] Initialization Completed successfully.");
            ctx.postMessage({ type: 'init-complete' });

        } else if (msg.type === 'release') {
            if (engine) {
                console.log("[AI Worker] Releasing engine memory...");
                engine.dispose();
                engine = null;
            }
            ctx.postMessage({ type: 'released' });

        } else if (msg.type === 'reinit') {
            // [Lock] If already initializing, just wait!
            if (initPromise) {
                console.log("[AI Worker] Already initializing, waiting...");
                await initPromise;
                ctx.postMessage({ type: 'init-complete' });
                return;
            }

            const config = (self as any).aiConfig;
            if (!config) {
                 ctx.postMessage({ type: 'error', message: 'No cached config for reinit' });
                 return;
            }
            
            if (config.onlyRules) {
                 console.log("[AI Worker] Re-Initialized (Rule-only Mode)");
                 ctx.postMessage({ type: 'init-complete' });
                 return;
            }

            if (!engine) {
                console.log("[AI Worker] Re-Initializing engine...");
                engine = new OnnxEngine({
                    modelPath: config.modelPath,
                    modelParts: config.modelParts,
                    wasmPath: config.wasmPath,
                    numThreads: config.numThreads,
                    debug: true
                });
                
                clearWatchdog();
                initWatchdog = setTimeout(() => {
                    ctx.postMessage({ type: 'error', message: 'Worker 重新初始化超时' });
                    initPromise = null;
                }, WATCHDOG_TIMEOUT);

                initPromise = engine.initialize((statusMsg) => {
                     // Be less verbose on re-init
                     if (statusMsg.includes('启动')) ctx.postMessage({ type: 'status', message: statusMsg });
                });
                await initPromise;
                initPromise = null;
                clearWatchdog();
            }
            // If engine exists and no promise, we assume it is ready.
            ctx.postMessage({ type: 'init-complete' });

        } else if (msg.type === 'compute') {
            const { board: boardState, history: gameHistory, color, size, gameType = 'Go', komi, difficulty, temperature } = msg.data;
            console.log(`[AI Worker] Compute Request Received. Type=${gameType}, Size=${size}, Diff=${difficulty}`); 

            // === Gomoku Logic ===
            if (gameType === 'Gomoku') {
                const board = boardState as BoardState;
                // [Fix] Defensive: ensure `size` matches actual board dimensions.
                // If they mismatch (e.g. stale boardSize from closure), use board.length.
                const safeSize = board.length;
                if (safeSize !== size) {
                    console.warn(`[AI Worker] Board size mismatch! size=${size}, board.length=${safeSize}. Using board.length.`);
                }
                const player = color;
                const opColor = player === 'black' ? 'white' : 'black';

                // 1. Initial Candidates & Safety Check
                // Fast path: if board is empty, play center
                let hasStone = false;
                for(let r=0; r<safeSize; r++) for(let c=0; c<safeSize; c++) if(board[r][c]) { hasStone = true; break; }
                if (!hasStone) {
                    const center = Math.floor(safeSize/2);
                    ctx.postMessage({ type: 'ai-response', data: { move: {x: center, y: center}, winRate: 0.5, lead: 0 } });
                    return;
                }

                // 2. Iterative Deepening Setup
                const isHard = difficulty === 'Hard';
                const isMedium = difficulty === 'Medium';
                
                let maxDepth = isHard ? 8 : (isMedium ? 4 : 2); // Depth limit
                // Time limit: prevent UI freeze (or pure worker lag)
                // Worker can run longer. 
                // Easy: 100ms, Medium: 800ms, Hard: 3000ms
                const timeLimit = isHard ? 3000 : (isMedium ? 800 : 100); 
                const startTime = performance.now();

                // Get Initial Candidates
                const candidates = getCandidateMoves(board, safeSize, 2);
                
                // Pre-Sort candidates by static score for Iterative Deepening efficiency
                // This gives us a good move ordering for Alpha-Beta
                const rootMoves = candidates.map(pt => ({
                    pt,
                    score: getGomokuScore(board, pt.x, pt.y, player, opColor, false)
                })).sort((a,b) => b.score - a.score);

                // Check Instant Win (Depth 0)
                if (rootMoves.length > 0 && rootMoves[0].score >= GOMOKU_SCORES.WIN) {
                     ctx.postMessage({ type: 'ai-response', data: { move: rootMoves[0].pt, winRate: 1.0, lead: 100 } });
                     return;
                }

                // Top K Pruning for Root
                const searchWidth = isHard ? 12 : (isMedium ? 8 : 5);
                const movesToSearch = rootMoves.slice(0, searchWidth).map(m => m.pt);

                let bestMove = movesToSearch[0];
                let currentBestScore = -Infinity;

                // --- Helper: Minimax (Local Recurse) ---
                const performSearch = (depth: number) => {
                    let alpha = -Infinity; // Root Alpha
                    const beta = Infinity;
                    let iterationBestMove = bestMove;
                    let iterationBestScore = -Infinity;

                    for (const move of movesToSearch) {
                        if (performance.now() - startTime > timeLimit) break;

                        // Do Move
                        board[move.y][move.x] = { color: player, x: move.x, y: move.y, id: 'sim' };
                        
                        // Recurse
                        // Next is Min (Opponent)
                        const score = minimaxGomokuRecursive(
                            board, depth - 1, alpha, beta, false, player, move
                        );

                        // Undo Move
                        board[move.y][move.x] = null;

                        if (score > iterationBestScore) {
                            iterationBestScore = score;
                            iterationBestMove = move;
                        }

                        // Alpha Update (Root)
                        if (score > alpha) {
                            alpha = score;
                        }
                        // No beta cutoff at root (we want to find best)
                    }
                    return { bestM: iterationBestMove, bestS: iterationBestScore };
                };
                
                // Iterative Deepening Loop
                for (let d = 2; d <= maxDepth; d += 2) {
                    const { bestM, bestS } = performSearch(d);
                    
                    // If we found a forced win, stop immediately
                    if (bestS >= GOMOKU_SCORES.WIN * 0.9) {
                        bestMove = bestM;
                        currentBestScore = bestS;
                        break;
                    }

                    if (performance.now() - startTime > timeLimit) {
                        // Don't update bestMove with partial search results if we timed out mid-iteration?
                        // Or trust the previous iteration.
                        // Ideally we only update if we finished the iteration or if the partial result is amazing.
                        // For simplicity, we just keep the previous completed iteration's best, 
                        // UNLESS we finished this iteration's loop?
                        // The loop above breaks if timeout.
                        // We should probably NOT update bestMove if d > 2 and we timed out early.
                        break; 
                    }
                    
                    bestMove = bestM;
                    currentBestScore = bestS;
                }

                // Add slight randomness for Easy/Medium to vary play?
                // Or deterministic high quality? User requested "Difficulty".
                // Keep it deterministic.

                ctx.postMessage({ 
                    type: 'ai-response', 
                    data: { 
                        move: bestMove, 
                        winRate: 0.5, // We don't have real winrate from heuristics
                        lead: 0 
                    } 
                });
                return;
            }
            
            // --- Blunder Logic (Artificial Stupidity for Easy/Medium) ---
            // We do this BEFORE the heavy AI engine to save performance and create "natural" mistakes.
            if ((gameType === 'Go' || gameType === undefined) && (difficulty === 'Easy' || difficulty === 'Medium')) {
                const blunderChance = difficulty === 'Easy' ? 0.20 : 0.05; // 20% for Easy, 5% for Medium
                console.log(`[AI Worker] Checking Blunder Logic. Chance=${blunderChance}, Random=${Math.random()}`);
                if (Math.random() < blunderChance) {
                    console.log(`[AI Worker] Triggering Blunder for ${difficulty} mode...`);
                    const board = boardState as BoardState;
                    const safeSize = board.length;
                    
                    // Generate random moves
                    const attempts = 50; 
                    for(let i=0; i<attempts; i++) {
                        const rx = Math.floor(Math.random() * safeSize);
                        const ry = Math.floor(Math.random() * safeSize);
                        
                        // Check basic legality (empty)
                        if (!board[ry][rx]) {
                             // Check suicide/ko rules using attemptMove
                             // We need a history logic? 
                             // For simplicity in blunder, we can just check if it's not suicide.
                             // We'll use the proper AttemptMove with empty history (or just prevHash if available) to be safe.
                             // But wait, `attemptMove` needs full board? Yes, we have `board`.
                             
                             let pHash: string | null = null;
                             if (gameHistory.length > 0) {
                                  const last = gameHistory[gameHistory.length-1];
                                  if (last && last.board) pHash = getBoardHash(last.board);
                             }

                             const isValid = attemptMove(board, rx, ry, color, 'Go', pHash);
                             if (isValid) {
                                  console.log(`[AI Worker] Blunder Move Selected: (${rx}, ${ry})`);
                                  ctx.postMessage({ 
                                      type: 'ai-response', 
                                      data: { 
                                          move: { x: rx, y: ry }, 
                                          winRate: 0.4, // Fake winrate
                                          lead: 0,
                                          ownership: []
                                      } 
                                  });
                                  return; // Stop here, skip engine!
                             }
                        }
                    }
                    console.log("[AI Worker] Failed to find valid blunder move, falling back to Engine.");
                }
            }

            // === Go Logic (Engine) ===
            if (!engine) {
                // [Fix] If engine is missing, we cannot analyze.
                // We should check if we can auto-recover or if we should fail.
                const config = (self as any).aiConfig;
                if (config && !config.onlyRules) {
                     console.warn("[AI Worker] Engine missing for compute. Attempting Auto-recovery...");
                     engine = new OnnxEngine({
                        modelPath: config.modelPath,
                        modelParts: config.modelParts,
                        wasmPath: config.wasmPath,
                        numThreads: config.numThreads,
                        debug: true
                    });
                    await engine.initialize();
                } else {
                    const mode = config?.onlyRules ? "Rule-only Mode" : "Engine NOT initialized";
                    throw new Error(`AI Engine unavailable (${mode}). Cannot compute move.`);
                }
            }

            const pla: Sign = color === 'black' ? 1 : -1;

            // 1. Reconstruct MicroBoard with Perfect Ko Detection
            // Logic: Replaying the entire history is the only way to ensure the internal 'ko' 
            // and group states of MicroBoard are perfectly synced. 
            // This is extremely fast (< 0.5ms for hundreds of moves).
            const board = new MicroBoard(size);
            const historyMoves: { color: Sign; x: number; y: number }[] = [];

            for (const item of gameHistory) {
                if (item.lastMove) {
                    const moveColor = item.currentPlayer === 'black' ? 1 : -1; 
                    // Use .play() to ensure captures and ko points are calculated
                    const ok = board.play(item.lastMove.x, item.lastMove.y, moveColor);
                    if (!ok) console.warn(`[AI Worker] Move replay failed: (${item.lastMove.x}, ${item.lastMove.y}) color=${moveColor}`);
                    
                    historyMoves.push({
                         color: moveColor,
                         x: item.lastMove.x,
                         y: item.lastMove.y
                    });
                } else {
                    // It was a PASS move in history
                    historyMoves.push({
                        color: item.currentPlayer === 'black' ? 1 : -1,
                        x: -1,
                        y: -1
                    });
                    // Reset ko on pass as per rules
                    board.ko = -1;
                }
            }
            
            // 3. Run Analysis
            console.log("[AI Worker] Calling engine.analyze...");
            const result = await engine.analyze(board, pla, {
                 history: historyMoves,
                 komi: komi ?? 7.5,
                 difficulty: difficulty,
                temperature: temperature
            });
            console.log("[AI Worker] Analysis returned.");

            // 4. Send Response
            
            // Check for Analysis Mode (No Move Selection)
            // @ts-ignore
            if (msg.data.mode === 'analyze') {
                 console.log("[AI Worker] Analysis Mode: Returning stats only.");
                 ctx.postMessage({
                    type: 'ai-response',
                    data: {
                        move: null, // No move
                        winRate: result.rootInfo.winrate,
                        lead: result.rootInfo.lead,
                        scoreStdev: result.rootInfo.scoreStdev,
                        ownership: result.rootInfo.ownership
                    }
                });
                return;
            }

            // Normal Move Selection
            let selectedMove: any = null;
            
            if (result.moves.length > 0) {
                 const validationBoard = boardState as BoardState;
                 
                 // Reconstruct prevHash (Simple Ko Check)
                 let prevHash: string | null = null;
                 if (gameHistory.length > 0) {
                     const lastItem = gameHistory[gameHistory.length - 1];
                     if (lastItem && lastItem.board) prevHash = getBoardHash(lastItem.board);
                 }

                 // Candidates list
                 let candidates = [...result.moves];

                 if (temperature && temperature > 0) {
                     // Weight-based Sampling (Retry loop for validity)
                     for (let retry = 0; retry < 20; retry++) {
                          if (candidates.length === 0) break;

                          let sumWeight = 0;
                          for (const m of candidates) sumWeight += (m as any).weight || m.prior;
                          
                          let r = Math.random() * sumWeight;
                          let pickedIndex = -1;
                          for (let i = 0; i < candidates.length; i++) {
                              const w = (candidates[i] as any).weight || candidates[i].prior;
                              r -= w;
                              if (r <= 0) { pickedIndex = i; break; }
                          }
                          if (pickedIndex === -1) pickedIndex = candidates.length - 1;

                          const candidate = candidates[pickedIndex];
                          if (candidate.x === -1) { selectedMove = null; break; } // Pass is valid
                          
                          if (attemptMove(validationBoard, candidate.x, candidate.y, color, 'Go', prevHash)) {
                               selectedMove = { x: candidate.x, y: candidate.y };
                               break;
                          } else {
                               candidates.splice(pickedIndex, 1); // Remove invalid
                          }
                     }
                     // Fallback to best valid if sampling failed
                     if (selectedMove === undefined) {
                         for (const m of result.moves) {
                             if (m.x === -1) { selectedMove = null; break; }
                             if (attemptMove(validationBoard, m.x, m.y, color, 'Go', prevHash)) {
                                 selectedMove = { x: m.x, y: m.y };
                                 break;
                             }
                         }
                     }
                 } else {
                     // Argmax (Iterate sorted)
                     for (const m of result.moves) {
                         if (m.x === -1) { selectedMove = null; break; }
                         if (attemptMove(validationBoard, m.x, m.y, color, 'Go', prevHash)) {
                             selectedMove = { x: m.x, y: m.y };
                             break;
                         }
                     }
                 }
            } else {
                selectedMove = null; // Pass
            }
            
            if (selectedMove === undefined) selectedMove = null; // Safety

            const isPass = selectedMove === null;
            // Only log if not null or valid object
            const moveStr = isPass ? 'Pass' : `(${selectedMove.x},${selectedMove.y})`;
            console.log(`[AI Worker] Best Move: ${moveStr} Win=${result.rootInfo.winrate.toFixed(1)}%`);

            ctx.postMessage({
                type: 'ai-response',
                data: {
                    move: selectedMove,
                    winRate: result.rootInfo.winrate,
                    lead: result.rootInfo.lead,
                    scoreStdev: result.rootInfo.scoreStdev,
                    ownership: result.rootInfo.ownership
                }
            });
        } else if (msg.type === 'stop') {
            // No-op for now as ONNX run is atomicish. 
            // We could set a flag if we had a loop.
        }
    } catch (err: any) {
        console.error('[AI Worker] Error:', err);
        // [Fix] Critical: If init failed, we must clear the engine instance so retry can work.
        // Otherwise 'reinit' thinks we are ready but session is null.
        if (engine) {
             console.error('[AI Worker] Resetting broken engine instance.');
             try { engine.dispose(); } catch (e) {}
             engine = null;
        }
        ctx.postMessage({ type: 'error', message: err.message });
    }
};

const minimaxGomokuRecursive = (
    board: BoardState, 
    depth: number, 
    alpha: number, 
    beta: number, 
    isMaximizing: boolean, 
    player: Player, 
    lastMove: Point | null
): number => {
    // Check Terminal (Win/Loss)
    if (lastMove && checkGomokuWin(board, lastMove)) {
        // If the *current* player just moved and won, that's great for them.
        // But minimax is called *after* the move.
        // So this means the PREVIOUS mover won. 
        // If isMaximizing=true, it means "Turn for Maximizer". 
        // So the previous mover was Minimizer. Minimizer won.
        // Return -Infinity
        return isMaximizing ? -100000000 : 100000000;
    }
    
    if (depth === 0) return 0;

    const size = board.length;
    // Optimization: Only search neighborhood of existing stones?
    // standard getCandidateMoves handles it (range=2)
    const candidates = getCandidateMoves(board, size, 2);
    if (candidates.length === 0) return 0;

    const myColor = player;
    const opColor = player === 'black' ? 'white' : 'black';
    // Current Mover Color
    const currentColor = isMaximizing ? player : opColor;
    // const nextColor    = isMaximizing ? opColor : player;
    
    // Heuristic Sort (Move Ordering)
    const scoredMoves = candidates.map(pt => {
        // Evaluate based on Current Mover's View
        const score = getGomokuScore(board, pt.x, pt.y, currentColor, isMaximizing ? opColor : player, false);
        return { pt, score };
    });
    
    scoredMoves.sort((a,b) => b.score - a.score);
    
    // Pruning
    const branching = depth > 2 ? 6 : 10;
    const movesToSearch = scoredMoves.slice(0, branching);

    if (isMaximizing) {
        let maxEval = -Infinity;
        for (const {pt} of movesToSearch) {
            // Check immediate win (Optimization)
            if (getGomokuScore(board, pt.x, pt.y, player, opColor, false) >= GOMOKU_SCORES.WIN) {
                return 100000000;
            }

            board[pt.y][pt.x] = { color: player, x: pt.x, y: pt.y, id: 'sim' };
            
            const evalScore = minimaxGomokuRecursive(board, depth - 1, alpha, beta, false, player, pt);
            
            board[pt.y][pt.x] = null; // Backtrack
            
            // Soft positional bonus
            const bonus = pt.x === Math.floor(size/2) && pt.y === Math.floor(size/2) ? 10 : 0;
            const total = evalScore + bonus * 0.01;
            
            maxEval = Math.max(maxEval, total);
            alpha = Math.max(alpha, total);
            if (beta <= alpha) break; 
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (const {pt} of movesToSearch) {
             // Check immediate win for Opponent (Optimization)
            if (getGomokuScore(board, pt.x, pt.y, opColor, player, false) >= GOMOKU_SCORES.WIN) {
                return -100000000;
            }

            board[pt.y][pt.x] = { color: opColor, x: pt.x, y: pt.y, id: 'sim' };
            
            const evalScore = minimaxGomokuRecursive(board, depth - 1, alpha, beta, true, player, pt);
            
            board[pt.y][pt.x] = null; // Backtrack
            
            minEval = Math.min(minEval, evalScore);
            beta = Math.min(beta, evalScore);
            if (beta <= alpha) break;
        }
        return minEval;
    }
};

export {};
