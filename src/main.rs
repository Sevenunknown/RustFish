use chess::{Board, ChessMove, Color, Game, Piece, BoardStatus, Rank, Square};
use macroquad::prelude::*;
use std::time::{Instant, Duration};
use std::collections::HashMap;
use std::sync::RwLock;
use once_cell::sync::Lazy;


// Chess AI trait
trait ChessAI {
    fn new(color: Color, name: String) -> Self where Self: Sized;
    fn get_move(&mut self, board: &Board, time_remaining: Duration) -> ChessMove;
    fn get_name(&self) -> &str;
    fn get_color(&self) -> Color;
    fn minmax(&self, board: &Board, depth: i32, alpha: f32, beta: f32) -> f32 {
        0.0
    }
    fn eval(&self, board: &Board) -> f32 {
        0.0
    }
    fn quiesence(&self, board:&Board) -> f32 {
        0.0
    }
}

// Random AI implementation
#[derive(Clone)]
struct RandomAI {
    color: Color,
    name: String,
}

struct BasicAI {
    color: Color,
    name: String
}

// --- Zobrist table (piece-square + side-to-move) ---
// Indexing: pieces[color_index][piece_index][square_index]
struct ZobristTable {
    pieces: [[[u64; 64]; 6]; 2],
    side: u64,
    // If you want to add castling/en-passant later, add fields here:
    castling: [u64; 16],
    en_passant: [u64; 8],
}

static ZOBRIST: Lazy<ZobristTable> = Lazy::new(|| {
    let seed: u64 = 123456789;
    rand::srand(seed); // seed once

    let mut pieces = [[[0u64; 64]; 6]; 2];
    for color in 0..2 {
        for piece in 0..6 {
            for sq in 0..64 {
                pieces[color][piece][sq] = rand::rand() as u64;
            }
        }
    }

    let mut castling = [0u64; 16];
    for i in 0..16 {
        castling[i] = rand::rand() as u64;
    }

    let mut en_passant = [0u64; 8];
    for i in 0..8 {
        en_passant[i] = rand::rand() as u64;
    }

    let side = rand::rand() as u64;

    ZobristTable {
        pieces,
        side: rand::rand() as u64,
        castling,
        en_passant,
    }
});


// Simple transposition table: hash -> (value, depth)
static TT: Lazy<RwLock<HashMap<u64, (f32, i32)>>> = Lazy::new(|| RwLock::new(HashMap::new()));

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

fn color_to_index(c: Color) -> usize {
    match c {
        Color::White => 0,
        Color::Black => 1,
    }
}

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
    // side to move
    if board.side_to_move() == Color::Black {
        h ^= ZOBRIST.side;
    }

    // TODO: include castling & en-passant:
    // h ^= ZOBRIST.castling[castling_index];
    // if let Some(file) = board.en_passant() { h ^= ZOBRIST.en_passant[file.to_index()]; }

    h
}

fn get_piece_vision(board: &Board, square: Square) -> chess::BitBoard {
    let piece = board.piece_on(square).unwrap();
    let color = board.color_on(square).unwrap();
    let vision = match piece {
        chess::Piece::Pawn => chess::get_pawn_moves(square, color, *board.combined()),
        chess::Piece::Bishop => chess::get_bishop_moves(square, *board.combined()),
        chess::Piece::Knight => chess::get_knight_moves(square) & !board.color_combined(color),
        chess::Piece::Rook => chess::get_rook_moves(square, *board.combined()),
        chess::Piece::Queen => chess::get_rook_moves(square, *board.combined()) | chess::get_bishop_moves(square, *board.combined()),
        chess::Piece::King => chess::get_king_moves(square),
    };
    vision
}

fn get_colors_vision(board: &Board, pieces: chess::BitBoard) -> chess::BitBoard {
    let mut vision = chess::BitBoard::new(0);
    for piece in pieces {
        vision = vision | get_piece_vision(&board, piece);
    }
    vision
}

fn lerp(v0: f32, v1: f32, t: f32) -> f32 {
    (1.0-t)*v0 + v1*t
}

impl ChessAI for RandomAI {
    fn new(color: Color, name: String) -> Self {
        RandomAI { color, name }
    }

    fn get_move(&mut self, board: &Board, _time_remaining: Duration) -> ChessMove {
        let moves: Vec<ChessMove> = chess::MoveGen::new_legal(board).collect();
        let idx = rand::gen_range(0, moves.len());
        moves[idx]
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_color(&self) -> Color {
        self.color
    }
}

impl ChessAI for BasicAI {
    fn new(color: Color, name: String) -> Self {
        BasicAI {color, name}
    }

    fn get_move(&mut self, board: &Board, time_remaining: Duration) -> ChessMove {
        let moves: Vec<ChessMove> = chess::MoveGen::new_legal(board).collect();
        // fallback if no moves (shouldn't happen in legal situations handled elsewhere)
        if moves.is_empty() {
            return ChessMove::new(Square::A1, Square::A1, None);
        }

        let mut best_move = moves[0];
        let mut best_score = f32::NEG_INFINITY;

        // iterative deepening settings
        let max_depth = 6; // you can increase this
        let start_time = Instant::now();
        let time_limit = {
            // give ourselves a bit of margin (e.g., 90% of remaining time for per-move)
            let millis = (time_remaining.as_millis() as f64 * 0.05).max(50.0);
            Duration::from_millis(millis as u64)
        };
        println!("{:?}", time_limit);

        // We'll start at 1 and go up to max_depth (or until time runs out)
        for depth in 3..=max_depth {
            // Check time
            if start_time.elapsed() > time_limit {
                break;
            }

            let mut local_best_move = best_move;
            let mut local_best_score = f32::NEG_INFINITY;
            // For move ordering: try previous best first if it's legal
            let mut ordered_moves = moves.clone();
            // move ordering heuristics could be added here

            for mv in &ordered_moves {
                // Handle pawn promotion (try to guess; prefer queen)
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

                // Use ChessMove::new_with_promotion when promotion is Some to be explicit
                let chess_move = if let Some(prom) = promotion {
                    //let promo_move = new_with_promotion(mv, Piece::Queen);
                    ChessMove::new(mv.get_source(), mv.get_dest(), Some(Piece::Queen))
                } else {
                    ChessMove::new(mv.get_source(), mv.get_dest(), None)
                };

                let new_board = board.make_move_new(chess_move);
                let score = -self.minmax(&new_board, depth - 1, f32::NEG_INFINITY, f32::INFINITY);
                if score > local_best_score {
                    local_best_score = score;
                    local_best_move = chess_move;
                }

                // time check inside loops to be responsive
                if start_time.elapsed() > time_limit {
                    break;
                }
            }

            // If we completed this depth's search, accept its best
            if start_time.elapsed() <= time_limit {
                best_move = local_best_move;
                best_score = local_best_score;
                // You can log depth completion
                println!("IterDepth {} -> best {} score {:.2}", depth, best_move, best_score);
            } else {
                // ran out of time during this depth; keep previous best
                println!("Time expired during depth {}", depth);
                break;
            }
        }

        println!("Selected move: {} with score {:.2}", best_move, best_score);
        best_move
    }

    fn minmax(&self, board: &Board, depth: i32, mut alpha: f32, beta: f32) -> f32 {
        // Transposition table probe
        let hash = compute_hash(board);
        if let Some((val, stored_depth)) = TT.read().unwrap().get(&hash).cloned() {
            if stored_depth >= depth {
                return val;
            }
        }

        match board.status() {
            BoardStatus::Checkmate => {
                // If side to move is checkmated, that's very bad for the side to move.
                // We return -inf from the perspective of the side who just moved; propagate signs from caller.
                return f32::NEG_INFINITY;
            }
            BoardStatus::Stalemate => return 0.0,
            _ => {}
        }

        if depth == 0 {
            let eval = self.quiesence(&board);
            // store
            TT.write().unwrap().insert(hash, (eval, depth));
            return eval;
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut moves = chess::MoveGen::new_legal(board);

        while let Some(mv) = moves.next() {
            let new_board = board.make_move_new(mv);
            let score = -self.minmax(&new_board, depth - 1, -beta, -alpha);

            if score > best_score {
                best_score = score;
            }

            if score > alpha {
                alpha = score;
            }

            if alpha >= beta {
                break; // alpha-beta cutoff
            }
        }

        // store result in TT
        TT.write().unwrap().insert(hash, (best_score, depth));
        best_score
    }

    fn quiesence(&self, board: &Board) -> f32 {
        let mut moves: Vec<ChessMove> = chess::MoveGen::new_legal(&board).collect();
        
        self.eval(&board)
    }

    fn eval(&self, board: &Board) -> f32 {
        let mut score = 0.0;
        let my_pieces = board.color_combined(self.color);
        let other_pieces = board.color_combined(!self.color);
        let my_vision = get_colors_vision(&board, *my_pieces);
        let other_vision = get_colors_vision(&board, *other_pieces);
        // Material evaluation
        for square in chess::ALL_SQUARES {
            if let Some(piece) = board.piece_on(square) {
                let color = board.color_on(square).unwrap();
                let value = match piece {
                    Piece::Pawn => 1.0,
                    Piece::Knight => 3.0,
                    Piece::Bishop => 3.1,
                    Piece::Rook => 5.0,
                    Piece::Queen => 9.0,
                    Piece::King => 0.0,
                };

                if color == self.color {
                    score += value;
                    if (chess::BitBoard::from_square(square) & other_vision).popcnt() > 0 {
                        score -= value*0.5;
                    }
                } else {
                    score -= value;
                    if (chess::BitBoard::from_square(square) & my_vision).popcnt() > 0 {
                        score += value*0.5;
                    }
                }
            }
        }
        // since all we did was update based on material we will give those more value
        // we will give the mobility less value
        score += (my_vision.popcnt() as f32 - other_vision.popcnt() as f32);
        // get a value between -1 and 1 for distance to the endgame (where -1 is the endgame and 1 isn't)
        let endgame_factor = lerp(-1.0, 1.0, board.combined().popcnt() as f32 / 32.0);
        // gets the number of moves the king has
        let my_king_mobility = (chess::get_king_moves(board.king_square(self.color)) & !other_vision & !my_pieces).popcnt();
        if my_king_mobility > 2 { // if it is greater than 2
            // it will lose score if the endgame factor is positive (we are in the middle to early game)
            // but near the endgame it will gain it
            score -= my_king_mobility as f32 * endgame_factor
        }
        if (chess::BitBoard::from_square(board.king_square(self.color)) & other_vision).popcnt() > 0 {
            score -= 5.0;
        }
        if (chess::BitBoard::from_square(board.king_square(!self.color)) & other_vision).popcnt() > 0 {
            score += 5.0;
        }
        if board.side_to_move() == self.color {
            score
        } else {
            -score
        }
    }

    fn get_name(&self) -> &str {
        &self.name
    }

    fn get_color(&self) -> Color {
        self.color
    }
}

// Piece textures manager (unchanged except minor fix: draw_texture_ex takes texture by value)
struct PieceTextures {
    texture: Texture2D,
    piece_size: f32,
}

impl PieceTextures {
    async fn new(image_path: &str) -> Self {
        let texture = load_texture(image_path).await.unwrap();
        texture.set_filter(FilterMode::Nearest);

        PieceTextures {
            texture,
            piece_size: 60.0,
        }
    }

    fn draw_piece(&self, piece: Piece, color: Color, x: f32, y: f32, size: f32) {
        let (row, col) = match (piece, color) {
            (Piece::Queen, Color::White) => (1, 0),
            (Piece::King, Color::White) => (1, 1),
            (Piece::Rook, Color::White) => (1, 2),
            (Piece::Knight, Color::White) => (1, 3),
            (Piece::Bishop, Color::White) => (1, 4),
            (Piece::Pawn, Color::White) => (1, 5),
            (Piece::Queen, Color::Black) => (0, 0),
            (Piece::King, Color::Black) => (0, 1),
            (Piece::Rook, Color::Black) => (0, 2),
            (Piece::Knight, Color::Black) => (0, 3),
            (Piece::Bishop, Color::Black) => (0, 4),
            (Piece::Pawn, Color::Black) => (0, 5),
        };

        let src_x = col as f32 * self.piece_size;
        let src_y = row as f32 * self.piece_size;

        draw_texture_ex(
            &self.texture,
            x,
            y,
            WHITE,
            DrawTextureParams {
                source: Some(Rect::new(src_x, src_y, self.piece_size, self.piece_size)),
                dest_size: Some(Vec2::new(size, size)),
                ..Default::default()
            },
        );
    }
}

struct ChessGame {
    game: Game,
    player_color: Color,
    white_ai: Option<Box<dyn ChessAI>>,
    black_ai: Option<Box<dyn ChessAI>>,
    selected_square: Option<chess::Square>,
    possible_moves: Vec<chess::Square>,
    white_time: Duration,
    black_time: Duration,
    game_over: bool,
    move_history: Vec<String>,
    piece_textures: PieceTextures,
    last_move_time: Instant,
}

impl ChessGame {
    fn new(
        player_color: Color,
        white_ai: Option<Box<dyn ChessAI>>,
        black_ai: Option<Box<dyn ChessAI>>,
        time_control: Duration,
        piece_textures: PieceTextures,
    ) -> Self {
        ChessGame {
            game: Game::new(),
            player_color,
            white_ai,
            black_ai,
            selected_square: None,
            possible_moves: Vec::new(),
            white_time: time_control,
            black_time: time_control,
            game_over: false,
            move_history: Vec::new(),
            piece_textures,
            last_move_time: Instant::now(),
        }
    }

    fn is_ai_turn(&self) -> bool {
        let turn = self.game.current_position().side_to_move();
        match turn {
            Color::White => self.white_ai.is_some() && self.player_color != Color::White,
            Color::Black => self.black_ai.is_some() && self.player_color != Color::Black,
        }
    }

    fn update_time(&mut self) {
        let elapsed = self.last_move_time.elapsed();
        match self.game.current_position().side_to_move() {
            Color::White => self.white_time = self.white_time.saturating_sub(elapsed),
            Color::Black => self.black_time = self.black_time.saturating_sub(elapsed),
        }
        self.last_move_time = Instant::now();
    }

    fn update_ai_move(&mut self) {
        if self.game_over {
            return;
        }

        self.update_time();

        let current_position = self.game.current_position();
        let turn = current_position.side_to_move();

        if let Some(ai) = match turn {
            Color::White => &mut self.white_ai,
            Color::Black => &mut self.black_ai,
        } {
            let time_remaining = match turn {
                Color::White => self.white_time,
                Color::Black => self.black_time,
            };

            let chess_move = ai.get_move(&current_position, time_remaining);
            self.make_move(chess_move);
        }
    }

    fn make_move(&mut self, chess_move: ChessMove) -> bool {
        self.update_time();
        if self.game.current_position().legal(chess_move) {
            self.game.make_move(chess_move);
            self.move_history.push(chess_move.to_string());
            self.selected_square = None;
            self.possible_moves.clear();

            let status = self.game.current_position().status();
            self.game_over = match status {
                BoardStatus::Checkmate => true,
                BoardStatus::Stalemate => true,
                _ => false,
            };
            true
        } else {
            false
        }
    }

    fn select_square(&mut self, square: chess::Square) {
        let board = self.game.current_position();
        if let Some(piece) = board.piece_on(square) {
            if board.color_on(square) == Some(self.player_color) {
                self.selected_square = Some(square);
                self.possible_moves = chess::MoveGen::new_legal(&board)
                    .filter(|m| m.get_source() == square)
                    .map(|m| m.get_dest())
                    .collect();
                return;
            }
        }

        if let Some(selected) = self.selected_square {
            if self.possible_moves.contains(&square) {
                // Handle pawn promotion
                let promotion = if let Some(Piece::Pawn) = board.piece_on(selected) {
                    let rank = square.get_rank();
                    // FIXED: White promotes on Rank::Eighth, Black promotes on Rank::First
                    if (rank == Rank::Eighth && board.side_to_move() == Color::White) ||
                        (rank == Rank::First && board.side_to_move() == Color::Black) {
                        Some(Piece::Queen)
                    } else {
                        None
                    }
                } else {
                    None
                };

                // Construct the move properly: use new_with_promotion if promotion is Some
                let chess_move = if let Some(prom) = promotion {
                    ChessMove::new(selected, square, Some(Piece::Queen))
                } else {
                    ChessMove::new(selected, square, None)
                };

                self.make_move(chess_move);
            } else {
                self.selected_square = None;
                self.possible_moves.clear();
            }
        }
    }
}

fn draw_board(
    board: &Board,
    selected_square: Option<chess::Square>,
    possible_moves: &[chess::Square],
    piece_textures: &PieceTextures,
) {
    let light_color = WHITE;
    let dark_color = DARKGREEN;
    let highlight_color = YELLOW;
    let move_color = GREEN;
    let capture_color = RED;
    let promotion_color = ORANGE; // Orange

    let square_size = screen_width().min(screen_height()) / 8.0;

    for rank in 0..8 {
        for file in 0..8 {
            let square = chess::Square::make_square(
                chess::Rank::from_index(7 - rank),
                chess::File::from_index(file)
            );
            let is_light = (rank + file) % 2 == 0;

            let color = if Some(square) == selected_square {
                highlight_color
            } else if possible_moves.contains(&square) {
                // Check if this is a promotion square
                if let Some(from) = selected_square {
                    if let Some(Piece::Pawn) = board.piece_on(from) {
                        let rank = square.get_rank();
                        // CONSISTENT: White promotes on Rank::Eighth, Black on Rank::First
                        if (rank == Rank::Eighth && board.side_to_move() == Color::White) ||
                           (rank == Rank::First && board.side_to_move() == Color::Black) {
                            promotion_color
                        } else if board.piece_on(square).is_some() {
                            capture_color
                        } else {
                            move_color
                        }
                    } else if board.piece_on(square).is_some() {
                        capture_color
                    } else {
                        move_color
                    }
                } else if board.piece_on(square).is_some() {
                    capture_color
                } else {
                    move_color
                }
            } else if is_light {
                light_color
            } else {
                dark_color
            };

            draw_rectangle(
                file as f32 * square_size,
                rank as f32 * square_size,
                square_size,
                square_size,
                color,
            );

            if let Some(piece) = board.piece_on(square) {
                let piece_color = board.color_on(square).unwrap();
                piece_textures.draw_piece(
                    piece,
                    piece_color,
                    file as f32 * square_size,
                    rank as f32 * square_size,
                    square_size,
                );
            }
        }
    }
}

fn format_time(duration: Duration) -> String {
    let secs = duration.as_secs();
    format!("{:02}:{:02}", secs / 60, secs % 60)
}

fn draw_ui(game: &ChessGame) {
    let square_size = screen_width().min(screen_height()) / 8.0;
    let ui_x = 8.0 * square_size + 20.0;

    let turn = game.game.current_position().side_to_move();
    let turn_text = format!("Turn: {}", if turn == Color::White { "White" } else { "Black" });
    draw_text(&turn_text, ui_x, 30.0, 20.0, WHITE);

    let white_time_color = if game.white_time < Duration::from_secs(30) { RED } else { WHITE };
    let black_time_color = if game.black_time < Duration::from_secs(30) { RED } else { WHITE };

    draw_text(&format!("White: {}", format_time(game.white_time)), ui_x, 60.0, 20.0, white_time_color);
    draw_text(&format!("Black: {}", format_time(game.black_time)), ui_x, 90.0, 20.0, black_time_color);

    draw_text("Move History:", ui_x, 130.0, 20.0, WHITE);
    for (i, mov) in game.move_history.iter().rev().take(10).enumerate() {
        draw_text(mov, ui_x, 160.0 + i as f32 * 25.0, 20.0, WHITE);
    }

    if game.game_over {
        let status = game.game.current_position().status();
        let message = match status {
            BoardStatus::Checkmate => format!("Checkmate! {} wins",
                if turn == Color::Black { "White" } else { "Black" }),
            BoardStatus::Stalemate => "Draw by stalemate".to_string(),
            _ => "Game over".to_string(),
        };
        draw_text(&message, ui_x, 400.0, 30.0, RED);
    }
}

#[macroquad::main("Chess Game")]
async fn main() {
    let piece_textures = PieceTextures::new("ChessPiecesArray.png").await;
    let player_color = Color::White;
    //let ai = BasicAI::new(Color::Black, "Chess AI".to_string());
    let ai2 = BasicAI::new(Color::White, "Chess AI 2".to_string());
    //let ai2 = None;
    let ai = RandomAI::new(Color::Black, "Random AI".to_string());
    let mut game = ChessGame::new(
        player_color,
        Some(Box::new(ai2)),
        Some(Box::new(ai)),
        Duration::from_secs(60 * 5), // 5 minutes per player
        piece_textures,
    );

    loop {
        let current_time = Instant::now();
        let elapsed = game.last_move_time.elapsed();
        match game.game.current_position().side_to_move() {
            Color::White => game.white_time = game.white_time.saturating_sub(elapsed),
            Color::Black => game.black_time = game.black_time.saturating_sub(elapsed),
        }
        game.last_move_time = current_time;

        if !game.game_over &&
            (game.white_time == Duration::from_secs(0) || game.black_time == Duration::from_secs(0)) {
            game.game_over = true;
        }

        if game.is_ai_turn() && !game.game_over {
            game.update_ai_move();
        }

        if is_mouse_button_pressed(MouseButton::Left) && !game.is_ai_turn() && !game.game_over {
            let square_size = screen_width().min(screen_height()) / 8.0;
            let mouse_pos = mouse_position();
            let file = (mouse_pos.0 / square_size) as usize;
            let rank = 7 - (mouse_pos.1 / square_size) as usize;

            if file < 8 && rank < 8 {
                let square = chess::Square::make_square(
                    chess::Rank::from_index(rank),
                    chess::File::from_index(file)
                );
                game.select_square(square);
            }
        }

        clear_background(BLACK);
        draw_board(
            &game.game.current_position(),
            game.selected_square,
            &game.possible_moves,
            &game.piece_textures,
        );
        draw_ui(&game);

        next_frame().await;
    }

    // Make a promotion move from an existing ChessMove + promotion piece
    fn new_with_promotion(mv: ChessMove, promotion: Piece) -> ChessMove {
        ChessMove::new(mv.get_source(), mv.get_dest(), Some(promotion))
    }

    // Make a promotion move directly from squares + promotion piece
    fn new_with_promotion_from_squares(src: Square, dst: Square, promotion: Piece) -> ChessMove {
        ChessMove::new(src, dst, Some(promotion))
    }
}
