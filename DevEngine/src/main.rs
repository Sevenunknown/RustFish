use std::{str::FromStr, time::{Duration, Instant}};
use chess::{Board, ChessMove, Color, Piece, BoardStatus, Square, Rank, File};
use std::io::{self, BufRead, Write};
use once_cell::sync::Lazy;
use dashmap::DashMap;

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
static TT: Lazy<DashMap<u64, (f32, i32, i32)>> = Lazy::new(DashMap::new);

#[inline(always)]
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

#[inline(always)]
fn color_to_index(c: Color) -> usize {
    match c {
        Color::White => 0,
        Color::Black => 1,
    }
}

#[inline]
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

#[inline]
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
    PawnEndgame {table: [i32; 64]},
    BishopEndgame {table: [i32; 64]},
    KnightEndgame {table: [i32; 64]},
    RookEndgame {table: [i32; 64]},
    QueenEndgame {table: [i32; 64]},
    KingEndgame {table: [i32; 64]},
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

    const PAWN_ENDGAME: Self = Self::PawnEndgame {table : [
            0,   0,   0,   0,   0,   0,   0,   0,
            178, 173, 158, 134, 147, 132, 165, 187,
            94, 100,  85,  67,  56,  53,  82,  84,
            32,  24,  13,   5,  -2,   4,  17,  17,
            13,   9,  -3,  -7,  -7,  -8,   3,  -1,
            4,   7,  -6,   1,   0,  -5,  -1,  -8,
            13,   8,   8,  10,  13,   0,   2,  -7,
            0,   0,   0,   0,   0,   0,   0,   0,]};

    const BISHOP: Self = Self::Bishop { table: [
            -29,   4, -82, -37, -25, -42,   7,  -8,
            -26,  16, -18, -13,  30,  59,  18, -47,
            -16,  37,  43,  40,  35,  50,  37,  -2,
            -4,   5,  19,  50,  37,  37,   7,  -2,
            -6,  13,  13,  26,  34,  12,  10,   4,
            0,  15,  15,  15,  14,  27,  18,  10,
            4,  15,  16,   0,   7,  21,  33,   1,
            -33,  -3, -14, -21, -13, -12, -39, -21,] };

    const BISHOP_ENDGAME: Self = Self::BishopEndgame { table : [
            -14, -21, -11,  -8, -7,  -9, -17, -24,
            -8,  -4,   7, -12, -3, -13,  -4, -14,
            2,  -8,   0,  -1, -2,   6,   0,   4,
            -3,   9,  12,   9, 14,  10,   3,   2,
            -6,   3,  13,  19,  7,  10,  -3,  -9,
            -12,  -3,   8,  10, 13,   3,  -7, -15,
            -14, -18,  -7,  -1,  4,  -9, -15, -27,
            -23,  -9, -23,  -5, -9, -16,  -5, -17,]};

    const KNIGHT: Self = Self::Knight { table: [
            -167, -89, -34, -49,  61, -97, -15, -107,
            -73, -41,  72,  36,  23,  62,   7,  -17,
            -47,  60,  37,  65,  84, 129,  73,   44,
            -9,  17,  19,  53,  37,  69,  18,   22,
            -13,   4,  16,  13,  28,  19,  21,   -8,
            -23,  -9,  12,  10,  19,  17,  25,  -16,
            -29, -53, -12,  -3,  -1,  18, -14,  -19,
            -105, -21, -58, -33, -17, -28, -19,  -23,] };

    const KNIGHT_ENDGAME: Self = Self::KnightEndgame { table: [
            -58, -38, -13, -28, -31, -27, -63, -99,
            -25,  -8, -25,  -2,  -9, -25, -24, -52,
            -24, -20,  10,   9,  -1,  -9, -19, -41,
            -17,   3,  22,  22,  22,  11,   8, -18,
            -18,  -6,  16,  25,  16,  17,   4, -18,
            -23,  -3,  -1,  15,  10,  -3, -20, -22,
            -42, -20, -10,  -5,  -2, -20, -23, -44,
            -29, -51, -23, -15, -22, -18, -50, -64,]};

    const ROOK: Self = Self::Rook { table: [
            32,  42,  32,  51, 63,  9,  31,  43,
            27,  32,  58,  62, 80, 67,  26,  44,
            -5,  19,  26,  36, 17, 45,  61,  16,
            -24, -11,   7,  26, 24, 35,  -8, -20,
            -36, -26, -12,  -1,  9, -7,   6, -23,
            -45, -25, -16, -17,  3,  0,  -5, -33,
            -44, -16, -20,  -9, -1, 11,  -6, -71,
            -19, -13,   1,  17, 16,  7, -37, -26,]};

    const ROOK_ENDGAME: Self = Self::RookEndgame {table: [
            13, 10, 18, 15, 12,  12,   8,   5,
            11, 13, 13, 11, -3,   3,   8,   3,
            7,  7,  7,  5,  4,  -3,  -5,  -3,
            4,  3, 13,  1,  2,   1,  -1,   2,
            3,  5,  8,  4, -5,  -6,  -8, -11,
            -4,  0, -5, -1, -7, -12,  -8, -16,
            -6, -6,  0,  2, -9,  -9, -11,  -3,
            -9,  2,  3, -1, -5, -13,   4, -20,]};

    const QUEEN: Self = Self::Queen { table: [  
            -28,   0,  29,  12,  59,  44,  43,  45,
            -24, -39,  -5,   1, -16,  57,  28,  54,
            -13, -17,   7,   8,  29,  56,  47,  57,
            -27, -27, -16, -16,  -1,  17,  -2,   1,
            -9, -26,  -9, -10,  -2,  -4,   3,  -3,
            -14,   2, -11,  -2,  -5,   2,  14,   5,
            -35,  -8,  11,   2,   8,  15,  -3,   1,
            -1, -18,  -9,  10, -15, -25, -31, -50,]};

    const QUEEN_ENDGAME: Self = Self::QueenEndgame { table : [
            -9,  22,  22,  27,  27,  19,  10,  20,
            -17,  20,  32,  41,  58,  25,  30,   0,
            -20,   6,   9,  49,  47,  35,  19,   9,
            3,  22,  24,  45,  57,  40,  57,  36,
            -18,  28,  19,  47,  31,  34,  39,  23,
            -16, -27,  15,   6,   9,  17,  10,   5,
            -22, -23, -30, -16, -16, -23, -36, -32,
            -33, -28, -22, -43,  -5, -32, -20, -41,]};

    const KING: Self = Self::King { table: [
            -65,  23,  16, -15, -56, -34,   2,  13,
            29,  -1, -20,  -7,  -8,  -4, -38, -29,
            -9,  24,   2, -16, -20,   6,  22, -22,
            -17, -20, -12, -27, -30, -25, -14, -36,
            -49,  -1, -27, -39, -46, -44, -33, -51,
            -14, -14, -22, -46, -44, -30, -15, -27,
            1,   7,  -8, -64, -43, -16,   9,   8,
            -15,  36,  12, -54,   8, -28,  24,  14,]};

    const KING_ENDGAME: Self = Self::KingEndgame {table: [
            -74, -35, -18, -18, -11,  15,   4, -17,
            -12,  17,  14,  17,  17,  38,  23,  11,
            10,  17,  23,  15,  20,  45,  44,  13,
            -8,  22,  24,  27,  26,  33,  26,   3,
            -18,  -4,  21,  24,  27,  23,   9, -11,
            -19,  -3,  11,  21,  23,  16,   7,  -9,
            -27, -11,   4,  13,  14,   4,  -5, -17,
            -53, -34, -21, -11, -28, -14, -24, -43]};

    pub fn get_value(&self, idx: usize) -> i32 {
        match self {
            PST::Pawn { table } => table[idx],
            PST::PawnEndgame { table } => table[idx],
            PST::Bishop { table } => table[idx],
            PST::BishopEndgame { table } => table[idx],
            PST::Knight { table } => table[idx],
            PST::KnightEndgame { table } => table[idx],
            PST::Rook { table } => table[idx],
            PST::RookEndgame { table } => table[idx],
            PST::Queen { table } => table[idx],
            PST::QueenEndgame { table } => table[idx],
            PST::King { table } => table[idx],
            PST::KingEndgame { table } => table[idx],
        }
    }
}

#[inline]
fn get_colors_vision(board: &Board, pieces: chess::BitBoard) -> chess::BitBoard {
    let mut vision = chess::BitBoard::new(0);
    for piece in pieces {
        vision = vision | get_piece_vision(board, piece);
    }
    vision
}

#[inline(always)]
fn lerp(v0: f32, v1: f32, t: f32) -> f32 {
    (1.0 - t) * v0 + t * v1
}

pub struct BasicAI {
    pub color: Color,
    pub name: String,
    pub start_time: Instant,
    pub info_time: Duration,
}

impl BasicAI {
    #[inline]
    pub fn new(color: Color, name: String) -> Self {
        BasicAI { color, name, start_time:Instant::now(), info_time:Duration::from_secs(1)}
    }

    pub fn get_move(&mut self, board: &Board, time_remaining: Duration) -> ChessMove {
        self.color = board.side_to_move();
        let moves: Vec<ChessMove> = chess::MoveGen::new_legal(board).collect();
        if moves.is_empty() {
            return ChessMove::new(Square::A1, Square::A1, None);
        }
        self.info_time = Duration::from_millis(50);
        let mut best_move = moves[0];
        let mut best_score = f32::NEG_INFINITY;
        let max_depth = 6;

        let time_limit = Duration::from_millis((time_remaining.as_millis() as f64 * 0.05).max(50.0) as u64);
        self.start_time = Instant::now();
        let mut depth_reached = 3;

        for depth in 1..=max_depth {
            if self.start_time.elapsed() > time_limit {
                break;
            }
            println!("info depth {depth} score cp {best_score}");
            depth_reached = depth;
            let mut local_best_move = best_move;
            let mut local_best_score = f32::NEG_INFINITY;
            
            for mv in &moves {
                if self.start_time.elapsed() > time_limit {
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

            if self.start_time.elapsed() <= time_limit {
                best_move = local_best_move;
                best_score = local_best_score;
            }
        }
        best_move
    }

    fn minmax(&self, board: &Board, depth: i32, mut alpha: f32, beta: f32) -> f32 {
        let hash = compute_hash(board);
        let mut flag = 2;
        

        if let Some(val) = TT.get(&hash) {
            let eval = val.0;
            
            if val.1 >= depth {
                let flag = val.2;
                let eval = val.0;
                if flag == 1 { return val.0 }
                else if flag == 3 && eval >= beta { return val.0 }
                else if flag == 2 && eval <= alpha { return val.0 }
            }
        }

        match board.status() {
            BoardStatus::Checkmate => return f32::NEG_INFINITY,
            BoardStatus::Stalemate => return 0.0,
            _ => {}
        }

        if depth == 0 {
            let eval = self.quiesence(board);
            TT.insert(hash, (eval, depth, 1));
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
                flag = 1;
                alpha = score;
            }

            if score >= beta {
                println!("info currmove {mv} score lowerbound {beta} upperbound {alpha} depth {depth}");
                flag = 3;
                break;
            }
        }

        TT.insert(hash, (best_score, depth, flag));
        best_score
    }

    #[inline]
    fn quiesence(&self, board: &Board) -> f32 {
        self.eval(board)
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
    let endgame_value = (board.combined()).popcnt() as f32 / 32.0;
    if piece == Piece::Pawn {
        value += 1.0;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
        value += lerp(PST::PAWN.get_value(idx) as f32 , PST::PAWN_ENDGAME.get_value(idx) as f32, endgame_value)/100.0
    }
    else if piece == Piece::Bishop {
        value += 3.1;
        value += get_piece_vision(board, square).popcnt() as f32;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
        value += lerp(PST::BISHOP.get_value(idx) as f32 , PST::BISHOP_ENDGAME.get_value(idx) as f32, endgame_value)/100.0
    }
    else if piece == Piece::Knight {
        value += 3.0;
        value += get_piece_vision(board, square).popcnt() as f32;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
        value += lerp(PST::KNIGHT.get_value(idx) as f32 , PST::KNIGHT_ENDGAME.get_value(idx) as f32, endgame_value)/100.0
    }
    else if piece == Piece::Rook {
        value += 5.0;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
        value += lerp(PST::ROOK.get_value(idx) as f32 , PST::ROOK_ENDGAME.get_value(idx) as f32, endgame_value)/100.0
    }
    else if piece == Piece::Queen {
        value += 9.0;
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
        value += lerp(PST::QUEEN.get_value(idx) as f32 , PST::QUEEN_ENDGAME.get_value(idx) as f32, endgame_value)/10.0
    }
    else {
        let idx = if color == Color::Black {63 - square.to_index()} else {square.to_index()};
        value += lerp(PST::KING.get_value(idx) as f32 , PST::KING_ENDGAME.get_value(idx) as f32, endgame_value)/100.0
    }

    value
}

fn need_info() {
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.is_empty() {
            continue;
        }
    }
}

// UCI Implementation
fn main() {
    let mut board = Board::default();
    let mut ai = BasicAI::new(Color::White, "Rusty".to_string());
    let stdin = io::stdin();

    println!("Rust Chess Engine ready");
    
    for line in stdin.lock().lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "uci" => {
                println!("id name RustyChess");
                println!("id author YourName");
                println!("option name Hash type spin default 64 min 1 max 1024");
                println!("option name Threads type spin default 1 min 1 max 4");
                println!("uciok");
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                board = Board::default();
            }
            "position" => {
                if parts.len() < 2 {
                    eprintln!("info string Invalid position command");
                    continue;
                }

                let mut idx = 1;
                let mut fen = String::new();
                
                // Parse position specification
                match parts[idx] {
                    "startpos" => {
                        board = Board::default();
                        idx += 1;
                    }
                    "fen" => {
                        idx += 1;
                        while idx < parts.len() && parts[idx] != "moves" {
                            if !fen.is_empty() {
                                fen.push(' ');
                            }
                            fen.push_str(parts[idx]);
                            idx += 1;
                        }
                        
                        match Board::from_fen(fen.clone()) {
                            Some(new_board) => board = new_board,
                            None => {
                                eprintln!("info string Invalid FEN: {}", fen);
                                board = Board::default();
                                continue;
                            }
                        }
                    }
                    _ => {
                        eprintln!("info string Invalid position specification");
                        continue;
                    }
                }

                // Parse moves if present
                if idx < parts.len() && parts[idx] == "moves" {
                    idx += 1;
                    for move_str in &parts[idx..] {
                        // Try UCI format first (e.g. "g1f3")
                        if move_str.len() == 4 || move_str.len() == 5 {
                            if let (Ok(src), Ok(dest)) = (
                                Square::from_str(&move_str[0..2]),
                                Square::from_str(&move_str[2..4]),
                            ) {
                                let promotion = if move_str.len() == 5 {
                                    match move_str.chars().nth(4).unwrap() {
                                        'q' => Some(Piece::Queen),
                                        'r' => Some(Piece::Rook),
                                        'b' => Some(Piece::Bishop),
                                        'n' => Some(Piece::Knight),
                                        _ => None,
                                    }
                                } else {
                                    None
                                };
                                
                                let chess_move = ChessMove::new(src, dest, promotion);
                                if board.legal(chess_move) {
                                    board = board.make_move_new(chess_move);
                                    continue;
                                }
                            }
                        }
                        
                        // Fall back to SAN parsing
                        match ChessMove::from_san(&board, move_str) {
                            Ok(chess_move) => {
                                if board.legal(chess_move) {
                                    board = board.make_move_new(chess_move);
                                } else {
                                    eprintln!("info string Illegal move: {}", chess_move);
                                }
                            }
                            Err(e) => eprintln!("info string Invalid move format '{}': {}", move_str, e),
                        }
                    }
                }
                println!("{board}");
            }
            "go" => {
                let mut time_left = Duration::from_secs(60);
                let mut time_per_move = Duration::from_millis(2000);
                
                for i in 1..parts.len() {
                    if parts[i] == "wtime" && i+1 < parts.len() {
                        if let Ok(time) = parts[i+1].parse::<u64>() {
                            time_left = Duration::from_millis(time);
                        }
                    } else if parts[i] == "btime" && i+1 < parts.len() {
                        if let Ok(time) = parts[i+1].parse::<u64>() {
                            time_left = Duration::from_millis(time);
                        }
                    } else if parts[i] == "movetime" && i+1 < parts.len() {
                        if let Ok(time) = parts[i+1].parse::<u64>() {
                            time_per_move = Duration::from_millis(time);
                        }
                    }
                }
                
                ai.color = board.side_to_move();
                println!("Board: {board}");
                let best_move = ai.get_move(&board, time_left.min(time_per_move));
                println!("bestmove {}", best_move.to_string());
            }
            "quit" => {
                break;
            }
            _ => {}
        }
        io::stdout().flush().unwrap();
    }
}

