use chess::{Board, ChessMove, MoveGen, Color, Square, Piece, BitBoard};
use rayon::prelude::*;
use core::f32;
use std::collections::HashMap;
use std::sync::RwLock;
use once_cell::sync::Lazy;

// Transposition Table Entry
#[derive(Clone, Copy)]
struct TTEntry {
    eval: f32,
    depth: i32,
}

// Global transposition table with hash keys
static TT: Lazy<RwLock<HashMap<u64, TTEntry>>> = Lazy::new(|| RwLock::new(HashMap::new()));

pub struct Engine {
    pub color: Color,
    pub max_depth: i32,
    eval: f32,
    has_castled: bool,
    previous_good_moves: Option<Vec<ChessMove>>,
}

impl Engine {
    const PAWN_TABLE: [i32; 64] =  
    [0,   0,   0,   0,   0,   0,  0,   0,
    98, 134,  61,  95,  68, 126, 34, -11,
    -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
    0,   0,   0,   0,   0,   0,  0,   0,];
    const KNIGHT_TABLE: [i32; 64] = [
        -167, -89, -34, -49,  61, -97, -15, -107,
        -73, -41,  72,  36,  23,  62,   7,  -17,
        -47,  60,  37,  65,  84, 129,  73,   44,
        -9,  17,  19,  53,  37,  69,  18,   22,
        -13,   4,  16,  13,  28,  19,  21,   -8,
        -23,  -9,  12,  10,  19,  17,  25,  -16,
        -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,];
    const BISHOP_TABLE: [i32; 64] = [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
        0,  15,  15,  15,  14,  27,  18,  10,
        4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ];
    const ROOK_TABLE: [i32; 64] = [
        32,  42,  32,  51, 63,  9,  31,  43,
        27,  32,  58,  62, 80, 67,  26,  44,
        -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,];
    const QUEEN_TABLE: [i32; 64] = [    
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
        -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
        -1, -18,  -9,  10, -15, -25, -31, -50,];
    const KING_TABLE: [i32; 64] = [    
        -65,  23,  16, -15, -56, -34,   2,  13,
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,];
    pub fn new(color: Color, max_depth: i32) -> Self {
        Self {
            color,
            max_depth,
            eval: 0.0,
            has_castled: false,
            previous_good_moves: None
        }
    }

    pub fn get_current_eval(&self) -> f32 {
        self.eval
    }
    // change
    pub fn get_move(&mut self, board: &Board) -> Option<ChessMove> {
        let legal_moves: Vec<_> = MoveGen::new_legal(board).collect();
        if legal_moves.is_empty() {
            self.eval = self.eval(board);
            return None;
        }
        let depth = self.lerp(self.max_depth as f32, 2.0 * self.max_depth as f32, 1.0 - board.combined().popcnt() as f32 / 32.0) as i32;
        let results: Vec<_> = legal_moves
            .par_iter()
            .map(|mv| {
                let new_board = board.make_move_new(*mv);
                let eval = -self.negamax(&new_board, depth-1, f32::NEG_INFINITY, f32::INFINITY);
                //println!("Finished {:?} doing {mv} with eval: {eval}", board.piece_on(mv.get_source()));
                (*mv, eval)
            })
            .collect();
        for (mv, eval) in &results {
            println!("Move: {mv}, Eval: {eval}");
        }
        if let Some((best_move, best_eval)) = results
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .copied()
        {
            self.eval = best_eval;
            let mut top_moves = results.clone();
            top_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            self.previous_good_moves = Some(top_moves.iter().take(50).map(|(m, _)| *m).collect());
            Some(best_move)
        } else {
            self.eval = self.eval(board);
            None
        }
    }

    fn negamax(&self, board: &Board, depth: i32,mut alpha: f32, beta: f32) -> f32 {
        // get the ordering of legal moves
        //TODO Add logic to the ordering
        let mut legal_moves = self.get_ordered_legal_moves(&board);
        // if the depth is 0 or no legal moves it returns 0 for now (it will switch to a quicense search when I make one)
    if legal_moves.next().is_none() {
        if board.checkers().popcnt() > 0 {
            return -100_000_000.00 + depth as f32; // mate in X
        } else {
            return 0.0; // stalemate
        }
    }
        if depth <= 0{
            return self.quiescence(&board, alpha, beta)
        }
        // sets the default max to -infinity
        let mut max = f32::NEG_INFINITY;
        // goes through every move
        for mv in legal_moves {
            let score = -self.negamax(&board.make_move_new(mv), depth-1, -alpha, -beta);
            max = max.max(score);
            alpha = alpha.max(score);
            if alpha >= beta {
                break;
            }
        }
        max
    }
    
    fn quiescence(&self, board: &Board, mut alpha: f32, beta:f32) -> f32 {
        //TODO Add an eval here
        let mut best_value = self.eval(board);
        if best_value >= beta {return best_value}
        if best_value > alpha {alpha = best_value}
        // gets a list of the captures
        // just grabs all the checks
        // Generate all tactical or threat-resolving moves
        let moves = MoveGen::new_legal(board)
            .filter(|mv| {
                // capture
                board.piece_on(mv.get_dest()).is_some() ||
                // gives check
                board.make_move_new(*mv).checkers().popcnt() > 0
                // resolves a hanging piece
                //(self.attacked_squares(&board.make_move_new(*mv), !board.side_to_move()) & board.color_combined(board.side_to_move())).popcnt() > 0
            });
        // goes through the moves
        for mv in moves {
            // uses a negamax idea for checking the next move if it is a capture or check
            let score = -self.quiescence(&board.make_move_new(mv), -beta, -alpha);
            // if score is better than beta just return it
            if score >= beta {
                return score;
            }
            // if the score is better than our best value set the best value
            if score >=  best_value {
                best_value = score;
            }
            // similarly with alpha
            if score > alpha {
                alpha = score;
            }
        }
        best_value
    }

    fn eval(&self, board: &Board) -> f32{
        // gets a bitboard of the pieces
        let white_pieces = board.color_combined(Color::White);
        let black_pieces = board.color_combined(Color::Black);
        // gets a bitboard of attacked pieces
        let white_sight = self.attacked_squares(&board, Color::White);
        let black_sight = self.attacked_squares(&board, Color::Black);
        // sets up the material scores
        let mut white_material_score = 0.0;
        let mut black_material_score = 0.0;
        for sq in white_pieces.into_iter() {
            let piece = board.piece_on(sq).unwrap();
            white_material_score += self.get_piece_value(piece, Color::White, sq, black_pieces.to_owned(), white_pieces.to_owned(), black_sight.to_owned(), white_sight.to_owned());
        }
        for sq in black_pieces.into_iter() {
            let piece = board.piece_on(sq).unwrap();
            black_material_score += self.get_piece_value(piece, Color::Black, sq, black_pieces.to_owned(), white_pieces.to_owned(), black_sight.to_owned(), white_sight.to_owned());
        }

        let score = white_material_score - black_material_score;
        score * if board.side_to_move() == Color::White {1.0} else {-1.0}
    }
    
    /*
    mobility (bonuses for pieces based on how many squares they're able to move to)
    king safety (a whole class of techniques based on giving penalties for the king being in danger or vice versa)
    pawn structure (a whole class of techniques, somewhat self explanatory)
    stuff like passed pawns,
    isolated pawns
    backwards pawns
    tempo (a simple static bonus for being the side to move)
    
    then some somewhat more specific stuff like bonuses for rooks on open files
    penalties for having pieces hanging or pinned 
    rooks on the seventh rank
    connected rooks
    knight outposts
     */
    fn get_piece_value(&self, piece: Piece, piece_color: Color, 
        square: Square, 
        black_pieces: BitBoard, white_pieces: BitBoard, 
        black_sight: BitBoard, white_sight: BitBoard) -> f32 {
        let index = match piece_color {
            Color::White => square.to_index(),
            Color::Black => 63-square.to_index()
        };

        
        let mut value = self.piece_value_standard(piece);
        if piece == Piece::Pawn {
            value += 1.0 + Engine::PAWN_TABLE[index] as f32 /10.0;
            let attacked = self.bitboard_contains(if piece_color == Color::White {black_sight} else {white_sight}, square);
            let defended = self.bitboard_contains(if piece_color == Color::White {white_sight} else {black_sight}, square);
            if defended && !attacked{
                value += 2.0;
            }
            else if defended && attacked {
                value += 5.0;
            }
            else if !defended && attacked {
                value -= 5.0;
            } 
            else { // if not attacked or defended
                value -= 1.0;
            }
        }
        else if piece == Piece::Rook {
            value += 5.0 + Engine::ROOK_TABLE[index] as f32 /10.0;
            let attacked = self.bitboard_contains(if piece_color == Color::White {black_sight} else {white_sight}, square);
            let defended = self.bitboard_contains(if piece_color == Color::White {white_sight} else {black_sight}, square);
            if defended && !attacked{
                value += 5.0;
            }
            else if defended && attacked {
                value -= 3.0;
            }
            else if !defended && attacked {
                value -= 10.0;
            } 
            else { // if not attacked or defended
                value -= 2.0;
            }
        }
        else if piece == Piece::Queen {
            value += 9.0 + Engine::QUEEN_TABLE[index] as f32 /10.0;
            let attacked = self.bitboard_contains(if piece_color == Color::White {black_sight} else {white_sight}, square);
            let defended = self.bitboard_contains(if piece_color == Color::White {white_sight} else {black_sight}, square);
            if defended && !attacked{
                value += 2.0;
            }
            else if defended && attacked {
                value -= 10.0;
            }
            else if !defended && attacked {
                value -= 15.0;
            } 
            else { // if not attacked or defended
                value += 0.0;
            }
        }
        else if piece == Piece::Knight {
            value += 3.0 + Engine::KNIGHT_TABLE[index] as f32 /10.0;
            let attacked = self.bitboard_contains(if piece_color == Color::White {black_sight} else {white_sight}, square);
            let defended = self.bitboard_contains(if piece_color == Color::White {white_sight} else {black_sight}, square);
            if defended && !attacked{
                value += 3.0;
            }
            else if defended && attacked {
                value -= 1.0;
            }
            else if !defended && attacked {
                value -= 5.0;
            } 
            else { // if not attacked or defended
                value += 1.0;
            }
        }
        else if piece == Piece::Bishop {
            value += 3.1 + Engine::BISHOP_TABLE[index] as f32 /10.0;
            let attacked = self.bitboard_contains(if piece_color == Color::White {black_sight} else {white_sight}, square);
            let defended = self.bitboard_contains(if piece_color == Color::White {white_sight} else {black_sight}, square);
            if defended && !attacked{
                value += 3.0;
            }
            else if defended && attacked {
                value -= 1.0;
            }
            else if !defended && attacked {
                value -= 5.0;
            } 
            else { // if not attacked or defended
                value += 1.0;
            }
        }
        else if piece == Piece::King {
            value += Engine::KING_TABLE[index] as f32 /10.0;
        }
        else {
            eprintln!("Error, Given piece is not a piece... Duh");
            return 0.0;
        }
        value
    }

    fn piece_value_standard(&self, piece: Piece) -> f32{
        match piece {
            Piece::Pawn => 1.0,
            Piece::Bishop => 3.1,
            Piece::Knight => 3.0,
            Piece::Rook => 5.0,
            Piece::Queen => 9.0,
            Piece::King => 0.0,
        }
    }

    fn attacked_squares(&self, board: &Board, color: Color) -> BitBoard {
        let mut attacks = BitBoard(0);
        let occupied = board.combined().to_owned();
        
        // Pawn attacks
        let pawns = board.pieces(Piece::Pawn) & board.color_combined(color);
        let pawns_u64 = pawns.0;
        
        if color == Color::White {
            // White pawns attack up-left and up-right
            let a_file_mask = 0x0101010101010101;
            let h_file_mask = 0x8080808080808080;
            
            let up_left = BitBoard((pawns_u64 << 7) & !a_file_mask);
            let up_right = BitBoard((pawns_u64 << 9) & !h_file_mask);
            attacks = attacks | up_left | up_right;
        } else {
            // Black pawns attack down-left and down-right
            let a_file_mask = 0x0101010101010101;
            let h_file_mask = 0x8080808080808080;
            
            let down_left = BitBoard((pawns_u64 >> 9) & !a_file_mask);
            let down_right = BitBoard((pawns_u64 >> 7) & !h_file_mask);
            attacks = attacks | down_left | down_right;
        }
        
        // Knights
        for sq in board.pieces(Piece::Knight) & board.color_combined(color) {
            attacks = attacks | chess::get_knight_moves(sq);
        }
        
        // Bishops/Queens
        for sq in (board.pieces(Piece::Bishop) | board.pieces(Piece::Queen)) & board.color_combined(color) {
            attacks = attacks | chess::get_bishop_moves(sq, occupied);
        }
        
        // Rooks/Queens
        for sq in (board.pieces(Piece::Rook) | board.pieces(Piece::Queen)) & board.color_combined(color) {
            attacks = attacks | chess::get_rook_moves(sq, occupied);
        }
        
        // King
        if let sq = (board.pieces(Piece::King) & board.color_combined(color)).to_square() {
            attacks = attacks | chess::get_king_moves(sq);
        }
        
        attacks
    }
    
    fn file_to_bitboard(&self, file: chess::File) -> BitBoard {
        let mut bb = BitBoard(0);
        for rank in chess::ALL_RANKS {
            bb |= BitBoard::from_square(Square::make_square(rank, file));
        }
        bb
    }

    fn bitboard_contains(&self, bitboard: BitBoard, square: Square) -> bool {
        (bitboard.0 & (1 << square.to_index())) != 0
    }

    fn lerp(&self, v0: f32, v1: f32, t:f32) -> f32 {
        (1.0-t)*v0 + t*v1
    }

    fn get_ordered_legal_moves(&self, board: &Board) -> MoveGen {
        return MoveGen::new_legal(board)
    }

    pub fn set_colour(&mut self, color: Color) {
        self.color = color;
    }

}