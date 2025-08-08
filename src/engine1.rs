use chess::{Board, ChessMove, Color, Piece, BoardStatus, Rank, Square};
use std::time::Duration;
use once_cell::sync::Lazy;
use dashmap::DashMap;  // Added for better concurrent access

// --- Zobrist Hashing ---
struct ZobristTable {
    pieces: [[[u64; 64]; 6]; 2],
    side: u64,
    castling: [u64; 16],
    en_passant: [u64; 8],
}

static ZOBRIST: Lazy<ZobristTable> = Lazy::new(|| {
    let mut pieces = [[[0u64; 64]; 6]; 2];
    for color in 0..2 {
        for piece in 0..6 {
            for sq in 0..64 {
                pieces[color][piece][sq] = rand::random::<u64>();
            }
        }
    }

    let mut castling = [0u64; 16];
    for i in 0..16 {
        castling[i] = rand::random::<u64>();
    }

    let mut en_passant = [0u64; 8];
    for i in 0..8 {
        en_passant[i] = rand::random::<u64>();
    }

    ZobristTable {
        pieces,
        side: rand::random::<u64>(),
        castling,
        en_passant,
    }
});

// Replaced RwLock<HashMap> with DashMap for better performance
static TT: Lazy<DashMap<u64, (f32, i32)>> = Lazy::new(DashMap::new);

#[inline(always)]  // Critical performance function
fn piece_to_index(p: Piece) -> usize {
    match p {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

#[inline(always)]  // Critical performance function
fn color_to_index(c: Color) -> usize {
    match c {
        Color::White => 0,
        Color::Black => 1,
    }
}

#[inline]  // Important to inline this hash computation
fn compute_hash(board: &Board) -> u64 {
    let mut h: u64 = 0;
    for sq in chess::ALL_SQUARES {
        if let Some(piece) = board.piece_on(sq) {
            if let Some(color) = board.color_on(sq) {
                let cidx = color_to_index(color);
                let pidx = piece_to_index(piece);
                let sqi = sq.to_index();
                h ^= ZOBRIST.pieces[cidx][pidx][sqi];
            }
        }
    }
    if board.side_to_move() == Color::Black {
        h ^= ZOBRIST.side;
    }
    h
}

#[inline]  // Frequently called in evaluation
fn get_piece_vision(board: &Board, square: Square) -> chess::BitBoard {
    let piece = board.piece_on(square).unwrap();
    let color = board.color_on(square).unwrap();
    match piece {
        chess::Piece::Pawn => chess::get_pawn_moves(square, color, *board.combined()),
        chess::Piece::Bishop => chess::get_bishop_moves(square, *board.combined()),
        chess::Piece::Knight => chess::get_knight_moves(square) & !board.color_combined(color),
        chess::Piece::Rook => chess::get_rook_moves(square, *board.combined()),
        chess::Piece::Queen => chess::get_rook_moves(square, *board.combined()) | chess::get_bishop_moves(square, *board.combined()),
        chess::Piece::King => chess::get_king_moves(square),
    }
}

enum PST {
    Pawn {table: [i32; 64]},
    Bishop {table: [i32; 64]},
    Knight {table: [i32; 64]},
    Rook {table: [i32; 64]},
    Queen {table: [i32; 64]},
    King {table: [i32; 64]},
}

impl PST {
    const PAWN: Self =  Self::Pawn {table : [      
            0,   0,   0,   0,   0,   0,  0,   0,
            98, 134,  61,  95,  68, 126, 34, -11,
            -6,   7,  26,  31,  65,  56, 25, -20,
            -14,  13,   6,  21,  23,  12, 17, -23,
            -27,  -2,  -5,  12,  17,   6, 10, -25,
            -26,  -4,  -4, -10,   3,   3, 33, -12,
            -35,  -1, -20, -23, -15,  24, 38, -22,
            0,   0,   0,   0,   0,   0,  0,   0,] };
    const BISHOP: Self = Self::Bishop { table: [
            -29,   4, -82, -37, -25, -42,   7,  -8,
            -26,  16, -18, -13,  30,  59,  18, -47,
            -16,  37,  43,  40,  35,  50,  37,  -2,
            -4,   5,  19,  50,  37,  37,   7,  -2,
            -6,  13,  13,  26,  34,  12,  10,   4,
            0,  15,  15,  15,  14,  27,  18,  10,
            4,  15,  16,   0,   7,  21,  33,   1,
            -33,  -3, -14, -21, -13, -12, -39, -21,] };
    const KNIGHT: Self = Self::Knight { table: [
            -167, -89, -34, -49,  61, -97, -15, -107,
            -73, -41,  72,  36,  23,  62,   7,  -17,
            -47,  60,  37,  65,  84, 129,  73,   44,
            -9,  17,  19,  53,  37,  69,  18,   22,
            -13,   4,  16,  13,  28,  19,  21,   -8,
            -23,  -9,  12,  10,  19,  17,  25,  -16,
            -29, -53, -12,  -3,  -1,  18, -14,  -19,
            -105, -21, -58, -33, -17, -28, -19,  -23,] };
    const ROOK: Self = Self::Rook { table: [
            32,  42,  32,  51, 63,  9,  31,  43,
            27,  32,  58,  62, 80, 67,  26,  44,
            -5,  19,  26,  36, 17, 45,  61,  16,
            -24, -11,   7,  26, 24, 35,  -8, -20,
            -36, -26, -12,  -1,  9, -7,   6, -23,
            -45, -25, -16, -17,  3,  0,  -5, -33,
            -44, -16, -20,  -9, -1, 11,  -6, -71,
            -19, -13,   1,  17, 16,  7, -37, -26,]};
    const QUEEN: Self = Self::Queen { table: [  
            -28,   0,  29,  12,  59,  44,  43,  45,
            -24, -39,  -5,   1, -16,  57,  28,  54,
            -13, -17,   7,   8,  29,  56,  47,  57,
            -27, -27, -16, -16,  -1,  17,  -2,   1,
            -9, -26,  -9, -10,  -2,  -4,   3,  -3,
            -14,   2, -11,  -2,  -5,   2,  14,   5,
            -35,  -8,  11,   2,   8,  15,  -3,   1,
            -1, -18,  -9,  10, -15, -25, -31, -50,]};
    const KING: Self = Self::King { table: [
            -65,  23,  16, -15, -56, -34,   2,  13,
            29,  -1, -20,  -7,  -8,  -4, -38, -29,
            -9,  24,   2, -16, -20,   6,  22, -22,
            -17, -20, -12, -27, -30, -25, -14, -36,
            -49,  -1, -27, -39, -46, -44, -33, -51,
            -14, -14, -22, -46, -44, -30, -15, -27,
            1,   7,  -8, -64, -43, -16,   9,   8,
            -15,  36,  12, -54,   8, -28,  24,  14,]};
    pub fn get_value(&self, idx: usize) -> i32 {
        match self {
            PST::Pawn { table } => table[idx],
            PST::Bishop { table } => table[idx],
            PST::Knight { table } => table[idx],
            PST::Rook { table } => table[idx],
            PST::Queen { table } => table[idx],
            PST::King { table } => table[idx],
        }
    }
}

#[inline]  // Important for evaluation
fn get_colors_vision(board: &Board, pieces: chess::BitBoard) -> chess::BitBoard {
    let mut vision = chess::BitBoard::new(0);
    for piece in pieces {
        vision = vision | get_piece_vision(board, piece);
    }
    vision
}

#[inline(always)]  // Small math function
fn lerp(v0: f32, v1: f32, t: f32) -> f32 {
    (1.0 - t) * v0 + t * v1
}

pub trait ChessAI {
    fn new(color: Color, name: String) -> Self where Self: Sized;
    fn get_move(&mut self, board: &Board, time_remaining: Duration) -> ChessMove;
    fn get_name(&self) -> &str;
    fn get_color(&self) -> Color;
    
    #[inline]  // Default implementations can be inlined too
    fn minmax(&self, board: &Board, depth: i32, alpha: f32, beta: f32) -> f32 {
        0.0
    }
    #[inline]
    fn eval(&self, board: &Board) -> f32 {
        0.0
    }
    #[inline]
    fn quiesence(&self, board: &Board, mut alpha: f32, beta: f32) -> f32 {
        0.0
    }

}

pub struct BasicAI {
    pub color: Color,
    pub name: String,
}

impl BasicAI {
    #[inline]  // Constructor is called frequently
    pub fn new(color: Color, name: String) -> Self {
        BasicAI { color, name }
    }
}

impl ChessAI for BasicAI {
    #[inline]
    fn new(color: Color, name: String) -> Self {
        BasicAI::new(color, name)
    }

    fn get_move(&mut self, board: &Board, time_remaining: Duration) -> ChessMove {
        let moves: Vec<ChessMove> = chess::MoveGen::new_legal(board).collect();
        if moves.is_empty() {
            return ChessMove::new(Square::A1, Square::A1, None);
        }

        let mut best_move = moves[0];
        let mut best_score = f32::NEG_INFINITY;
        let max_depth = 6;

        let time_limit = Duration::from_millis((time_remaining.as_millis() as f64 * 0.07).max(30.0) as u64);
        println!("Time Limit: {:?}", time_limit);
        let start_time = std::time::Instant::now();
        let mut depth_reached = 3;  // Track actual depth reached
<<<<<<< Updated upstream
        
        for depth in 3..=max_depth {
=======
        println!("First Move: {}", moves[0]);
        for depth in 1..=max_depth {
>>>>>>> Stashed changes
            if start_time.elapsed() > time_limit {
                println!("Time expired at depth: {}", depth-1);
                break;
            }
            
            depth_reached = depth;
            let mut local_best_move = best_move;
<<<<<<< Updated upstream
            let mut local_best_score = f32::NEG_INFINITY;
            
=======
            let mut local_best_score = best_score;
>>>>>>> Stashed changes
            for mv in &moves {
                if start_time.elapsed() > time_limit {
                    break;
                }

                let promotion = if let Some(Piece::Pawn) = board.piece_on(mv.get_source()) {
                    let rank = mv.get_dest().get_rank();
                    if (rank == Rank::Eighth && board.side_to_move() == Color::White) ||
                        (rank == Rank::First && board.side_to_move() == Color::Black) {
                        Some(Piece::Queen)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let chess_move = if let Some(prom) = promotion {
                    ChessMove::new(mv.get_source(), mv.get_dest(), Some(prom))
                } else {
                    ChessMove::new(mv.get_source(), mv.get_dest(), None)
                };

                let new_board = board.make_move_new(chess_move);
                let score = -self.minmax(&new_board, depth - 1, f32::NEG_INFINITY, f32::INFINITY);
                
                if score > local_best_score {
                    local_best_score = score;
                    local_best_move = chess_move;
                }
            }

            if start_time.elapsed() <= time_limit {
                best_move = local_best_move;
                best_score = local_best_score;
            }
        }
        println!("Reached depth: {}", depth_reached);
        best_move
    }

    #[inline]
    fn get_name(&self) -> &str {
        &self.name
    }

    #[inline]
    fn get_color(&self) -> Color {
        self.color
    }

    fn minmax(&self, board: &Board, depth: i32, mut alpha: f32, beta: f32) -> f32 {
        let hash = compute_hash(board);
        
        // Faster DashMap access pattern
        if let Some(val) = TT.get(&hash) {
            if val.1 >= depth {
                return val.0;
            }
        }

        match board.status() {
            BoardStatus::Checkmate => if board.side_to_move() == self.color {
            return f32::NEG_INFINITY; // We're checkmated
            } else {
            return f32::INFINITY; // We checkmated opponent
            },
            BoardStatus::Stalemate => return 0.0,
            _ => {}
        }

        if depth == 0 {
<<<<<<< Updated upstream
            let eval = self.quiesence(board);
            TT.insert(hash, (eval, depth));
=======
            let eval = self.quiesence(board, alpha, beta);
            TT.insert(hash, (eval, depth, 1));
>>>>>>> Stashed changes
            return eval;
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut movegen = chess::MoveGen::new_legal(board);

        while let Some(mv) = movegen.next() {
            let new_board = board.make_move_new(mv);
            let score = -self.minmax(&new_board, depth - 1, -beta, -alpha);

            if score > best_score {
                best_score = score;
            }

            if score > alpha {
                alpha = score;
            }

            if alpha >= beta {
                break;
            }
        }

        TT.insert(hash, (best_score, depth));
        best_score
    }

    #[inline]
    fn quiesence(&self, board: &Board, mut alpha: f32, beta: f32) -> f32 {
        // Stand pat (current evaluation)
        let stand_pat = self.eval(board);
        if stand_pat >= beta {
            return beta;
        }
        alpha = alpha.max(stand_pat);
        if stand_pat < alpha - 9.0 { // if the difference is more than a queen's value (meaning the capture cannot be worth it)
            return alpha
        }
        // Only generate captures (no checks for now)
        let mut movegen = chess::MoveGen::new_legal(board);
        movegen.set_iterator_mask(*board.color_combined(!board.side_to_move()));

        // Consider each capture
        for mv in movegen {
            // Delta pruning - skip moves that can't possibly raise alpha
            let captured_value = match board.piece_on(mv.get_dest()) {
                Some(p) => [0.0, 1.0, 3.0, 3.1, 5.0, 9.0][piece_to_index(p)],
                None => continue, // Shouldn't happen with our mask
            };
            
            if stand_pat + captured_value + 0.5 < alpha {
                continue; // Can't possibly improve alpha
            }

            let new_board = board.make_move_new(mv);
            let score = -self.quiesence(&new_board, -beta, -alpha);

            if score >= beta {
                return beta;
            }
            if score > alpha {
                alpha = score;
            }
        }

        alpha
    }
        
    #[inline]
    fn eval(&self, board: &Board) -> f32 {
        let mut score = 0.0;
        let my_pieces = board.color_combined(self.color);
        let other_pieces = board.color_combined(!self.color);
        let my_vision = get_colors_vision(board, *my_pieces);
        let other_vision = get_colors_vision(board, *other_pieces);

        // Material evaluation
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let color = board.color_on(square).unwrap();
                let value = get_piece_value(&board, square);

                if color == self.color {
                    score += value;
                    if (chess::BitBoard::from_square(square) & other_vision).popcnt() > 0 {
                        score -= value * 0.5;
                    }
                } else {
                    score -= value;
                    if (chess::BitBoard::from_square(square) & my_vision).popcnt() > 0 {
                        score += value * 0.5;
                    }
                }
            }
        }

        // Mobility and king safety
        score += (my_vision.popcnt() as f32 - other_vision.popcnt() as f32) * 0.1;
        
        let endgame_factor = lerp(-1.0, 1.0, board.combined().popcnt() as f32 / 32.0);
        let my_king_mobility = (chess::get_king_moves(board.king_square(self.color)) & !other_vision & !my_pieces).popcnt();
        
        if my_king_mobility > 2 {
            score -= my_king_mobility as f32 * endgame_factor;
        }

        if (chess::BitBoard::from_square(board.king_square(self.color)) & other_vision).popcnt() > 0 {
            score -= 5.0;
        }

        if (chess::BitBoard::from_square(board.king_square(!self.color)) & my_vision).popcnt() > 0 {
            score += 5.0;
        }

        if board.side_to_move() == self.color {
            score
        } else {
            -score
        }
    }
}

fn get_piece_value(board: &Board, square: Square) -> f32 {
    let mut value = 0.0;
    let piece = board.piece_on(square).unwrap();
    let color = board.color_on(square).unwrap();
<<<<<<< Updated upstream
    if piece == Piece::Pawn {
        value += 1.0;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
        value += PST::PAWN.get_value(idx as usize) as f32 / 10.0
=======
    let endgame_value = 1.0-(board.combined()).popcnt() as f32 / 32.0;
    if piece == Piece::Pawn {
        value += 1.0;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
        value += lerp(PST::PAWN.get_value(idx) as f32 , PST::PAWN_ENDGAME.get_value(idx) as f32, endgame_value)
>>>>>>> Stashed changes
    }
    else if piece == Piece::Bishop {
        value += 3.1;
        value += get_piece_vision(&board, square).popcnt() as f32;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
<<<<<<< Updated upstream
        value += PST::BISHOP.get_value(idx as usize) as f32 / 10.0
=======
        value += lerp(PST::BISHOP.get_value(idx) as f32 , PST::BISHOP_ENDGAME.get_value(idx) as f32, endgame_value)
>>>>>>> Stashed changes
    }
    else if piece == Piece::Knight {
        value += 3.0;
        value += get_piece_vision(&board, square).popcnt() as f32;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
<<<<<<< Updated upstream
        value += PST::KNIGHT.get_value(idx as usize) as f32 / 10.0
=======
        value += lerp(PST::KNIGHT.get_value(idx) as f32 , PST::KNIGHT_ENDGAME.get_value(idx) as f32, endgame_value)
>>>>>>> Stashed changes
    }
    else if piece == Piece::Rook {
        value += 5.0;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
<<<<<<< Updated upstream
        value += PST::ROOK.get_value(idx as usize) as f32 / 10.0
=======
        value += lerp(PST::ROOK.get_value(idx) as f32 , PST::ROOK_ENDGAME.get_value(idx) as f32, endgame_value)
>>>>>>> Stashed changes
    }
    else if piece == Piece::Queen {
        value += 9.0;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
<<<<<<< Updated upstream
        value += PST::QUEEN.get_value(idx as usize) as f32 / 10.0
=======
        value += lerp(PST::QUEEN.get_value(idx) as f32 , PST::QUEEN_ENDGAME.get_value(idx) as f32, endgame_value)
>>>>>>> Stashed changes
    }
    else { // king
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
<<<<<<< Updated upstream
        value += PST::KING.get_value(idx as usize) as f32 / 10.0
=======
        value += lerp(PST::KING.get_value(idx) as f32 , PST::KING_ENDGAME.get_value(idx) as f32, endgame_value);
        value -= get_piece_vision(&board, square).popcnt() as f32;
>>>>>>> Stashed changes
    }

    value
}

fn see(board: &Board, mv: ChessMove) -> i32 {
    const PIECE_VALUES: [i32; 6] = [100, 300, 300, 500, 900, 0];
    
    let from = mv.get_source();
    let to = mv.get_dest();
    let attacker = board.piece_on(from).unwrap();
    let victim = board.piece_on(to);
    
    // Return 0 for non-captures
    let victim = match victim {
        Some(p) => p,
        None => return 0,
    };

    let mut gain = PIECE_VALUES[piece_to_index(victim)];
    let mut depth = 0;
    let mut side = board.side_to_move();
    let mut occupied = *board.combined();
    
    occupied ^= chess::BitBoard::from_square(from); // Remove attacker
    
    loop {
        let attackers = get_attackers(board, to, side) & occupied;
        if attackers == chess::BitBoard(0) {
            break;
        }
        
        // Find least valuable attacker
        let mut min_value = i32::MAX;
        let mut min_attacker = None;
        
        for sq in attackers {
            let piece = board.piece_on(sq).unwrap();
            let value = PIECE_VALUES[piece_to_index(piece)];
            if value < min_value {
                min_value = value;
                min_attacker = Some(sq);
            }
        }
        
        let attacker_sq = match min_attacker {
            Some(sq) => sq,
            None => break,
        };
        
        // Update gain and prepare next iteration
        if depth == 0 {
            gain -= min_value;
        } else {
            gain = (-gain).max(0) - min_value;
        }
        
        occupied ^= chess::BitBoard::from_square(attacker_sq);
        side = !side;
        depth += 1;
    }
    
    gain
}

fn get_attackers(board: &Board, square: Square, attacker_color: Color) -> chess::BitBoard {
    let occupied = *board.combined();
    let attackers = board.color_combined(attacker_color);
    
    // Pawn attacks
    let pawn_attacks = chess::get_pawn_attacks(square, board.color_on(square).unwrap(), board.combined().to_owned()) & 
                    board.pieces(Piece::Pawn) & attackers;
    
    // Knight attacks
    let knight_attacks = chess::get_knight_moves(square) & 
                        board.pieces(Piece::Knight) & attackers;
    
    // Bishop/Queen attacks
    let bishop_attacks = chess::get_bishop_moves(square, occupied) & 
                        (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen)) & attackers;
    
    // Rook/Queen attacks
    let rook_attacks = chess::get_rook_moves(square, occupied) & 
                    (board.pieces(Piece::Rook) | board.pieces(Piece::Queen)) & attackers;
    
    // King attacks
    let king_attacks = chess::get_king_moves(square) & 
                    board.pieces(Piece::King) & attackers;

    pawn_attacks | knight_attacks | bishop_attacks | rook_attacks | king_attacks
}