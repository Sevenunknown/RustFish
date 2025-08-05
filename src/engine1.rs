use chess::{Board, ChessMove, MoveGen, Color, Square, Piece, BitBoard};
use rayon::prelude::*;
use core::f32;
use std::collections::{HashMap, HashSet};
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
    const PAWN_TABLE: [[f32; 8]; 8] = [
        [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        [0.9,0.9,0.85,0.8,0.8,0.85,0.9,0.9],
        [0.85,0.85,0.8,0.8,0.8,0.8,0.85,0.85],
        [0.30,0.6,0.65,0.95,0.95,0.65,0.6,0.30],
        [0.30,0.6,0.85,0.95,0.95,0.85,0.6,0.30],
        [0.25,0.6,0.65,0.85,0.85,0.65,0.6,0.55],
        [0.25,0.25,0.25,0.25,0.25,0.25,0.25,0.25],
        [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        ];
    const KNIGHT_TABLE: [[f32; 8];8] = [
        [-0.50,-0.50,-0.50,-0.50,-0.50,-0.50,-0.50,-0.50],
        [-0.50,0.96,0.95,0.9,0.9,0.95,0.96,-0.50],
        [-0.50,0.90,0.92,0.95,0.95,0.92,0.90,-0.50],
        [-0.50,0.90,0.95,0.90,0.90,0.95,0.90,-0.50],
        [-0.50,0.90,0.96,0.90,0.90,0.96,0.90,-0.50],
        [-0.50,0.90,0.92,0.95,0.95,0.92,0.90,-0.50],
        [-0.50,0.96,0.95,0.9,0.9,0.95,0.96,-0.50],
        [-0.50,-0.50,-0.50,-0.50,-0.50,-0.50,-0.50,-0.50],
    ];
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
            let depth = self.lerp(0.0, self.max_depth as f32, 1.0-board.combined().popcnt() as f32/32.0) as i32 + self.max_depth;
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
                if self.is_castling_move(&board, best_move) {
                    self.has_castled = true;
                }
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
            if depth <= 0 || legal_moves.next().is_none(){
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
            let mut best_value = self.eval(&board);
            if best_value >= beta {return best_value}
            if best_value > alpha {alpha = best_value}
            // gets a list of the captures
            let mut captures: Vec<_> = MoveGen::new_legal(board)
                .filter(|mv| board.piece_on(mv.get_dest()).is_some())
                .collect();
            // sorts the captures using a MVV - LVA Model
            captures.sort_by(|a, b| {
                let v_a = board.piece_on(a.get_dest()).map_or(0, |p| self.get_piece_value(p) as i32);
                let v_b = board.piece_on(b.get_dest()).map_or(0, |p| self.get_piece_value(p) as i32);
                let a_a = board.piece_on(a.get_source()).map_or(0, |p| self.get_piece_value(p) as i32);
                let a_b = board.piece_on(b.get_source()).map_or(0, |p| self.get_piece_value(p) as i32);

                //MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                (v_b * 10 - a_b).cmp(&(v_a * 10 - a_a))
            });
            // just grabs all the checks
            let mut checks: Vec<_> = MoveGen::new_legal(board)
                .filter(|mv| board.make_move_new(mv.to_owned()).checkers().popcnt() > 0)
                .collect();
            // this will search checks first then captures because I suspect finding a checkmate would be better and there
            // are fewer checks than captures usually
            checks.extend(captures);
            // goes through the moves
            for mv in checks {
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
            let white_pieces = board.color_combined(Color::White);
            let black_pieces = board.color_combined(Color::Black);
            let mut white_material_score = 0.0;
            let mut black_material_score = 0.0;
            for piece in &[Piece::Pawn, Piece::Bishop,Piece::Knight,Piece::Rook,Piece::Queen] {
                // gets the count for both sides of the number of each piece they have
                let white_piece_count = (white_pieces & board.pieces(*piece)).popcnt() as f32;
                let black_piece_count = (black_pieces & board.pieces(*piece)).popcnt() as f32;
                white_material_score += white_piece_count*self.get_piece_value(*piece);
                black_material_score += black_piece_count*self.get_piece_value(*piece);
            }
            let score = white_material_score - black_material_score;
            score * if board.side_to_move() == Color::White {1.0} else {-1.0}
        }
        
        fn get_piece_value(&self, piece: Piece) -> f32 {
            match piece {
                Piece::Pawn => 1.0,
                Piece::Bishop => 3.1,
                Piece::Knight => 3.0,
                Piece::Rook => 5.0,
                Piece::Queen => 9.0,
                Piece::King => 0.0
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

    pub fn set_colour(&mut self, colour: Color) {
        self.color = colour;
    }

    fn manhattan_distance_with_squares(&self, square_1: Square, square_2: Square) -> f32 {
        // get the ranks as a value
        let rank_1 = square_1.get_rank().to_index() as f32;
        let rank_2 = square_2.get_rank().to_index() as f32;
        // get the files as a value
        let file_1 = square_1.get_file().to_index() as f32;
        let file_2 = square_2.get_file().to_index() as f32;
        // calculate the manhattan distance
        (rank_1-rank_2).abs() + (file_1-file_2).abs()
    }

    fn lerp(&self, v0: f32, v1: f32, t: f32) -> f32 {
        (1.0-t)*v0 + t*v1
    }
    
    fn is_castling_move(&self, board: &Board, mv: ChessMove) -> bool {
    // Additional verification
    let king_square = mv.get_source();
    let dest_square = mv.get_dest();
    
    // Must be king moving 2 squares horizontally
    board.piece_on(king_square) == Some(Piece::King) &&
    king_square.get_rank() == dest_square.get_rank() &&
    king_square.get_file().to_index().abs_diff(dest_square.get_file().to_index()) == 2
    }

    fn manhattan_distance_with_points(&self, x1: f32, y1: f32, x2:f32, y2:f32) -> f32 {
        (x1-x2).abs() + (y1-y2).abs()
    }

    fn has_castled_on_board(&self, board: &Board, color: Color) -> bool {
    let (king_sq, kingside_rook_sq, queenside_rook_sq) = match color {
        Color::White => (Square::G1, Square::F1, Square::D1),
        Color::Black => (Square::G8, Square::F8, Square::D8),
    };
    
    let king_pos = board.king_square(color);
    if king_pos == king_sq {
        if board.piece_on(kingside_rook_sq) == Some(Piece::Rook) {
            return true;
        }
    }
    if king_pos == Square::C1 || king_pos == Square::C8 {
        if board.piece_on(queenside_rook_sq) == Some(Piece::Rook) {
            return true;
        }
    }
    false
    }

    fn get_ordered_legal_moves(&self, board: &Board) -> MoveGen {
        return MoveGen::new_legal(board)
    }

    fn get_piece_location_value(&self, board: &Board, square: Square) -> f32{
        let piece = board.piece_on(square);
        let rank = if board.color_on(square).unwrap() ==  Color::White {square.get_rank().to_index()} else {7-square.get_rank().to_index()};
        let file = square.get_file().to_index();
        if piece.unwrap() == Piece::Pawn {
            let v = Engine::PAWN_TABLE[rank][file];
                return 1.0 + v
        }
        else if piece.unwrap() == Piece::Knight {
            let v = Engine::KNIGHT_TABLE[rank][file];
            return 1.0+v;
        }
        1.0
    }
}