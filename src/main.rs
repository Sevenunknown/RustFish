use chess::{Board, ChessMove, Color, Game, Piece, BoardStatus, Rank, Square};
use macroquad::prelude::*;
use std::time::{Instant, Duration};


// main.rs
mod engine1;  // Declare the engine1 module

use engine1::{ChessAI, BasicAI};  // Import the trait and struct






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
    white_ai: Option<BasicAI>,
    black_ai: Option<BasicAI>,
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
        white_ai: Option<BasicAI>,
        black_ai: Option<BasicAI>,
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
    let ai = BasicAI::new(Color::Black, "Chess AI".to_string());
    //let ai2 = BasicAI::new(Color::White, "Chess AI 2".to_string());
    //let ai2 = None;
    //let ai = RandomAI::new(Color::Black, "Random AI".to_string());
    let mut game = ChessGame::new(
        player_color,
        None,
        Some(ai),
        //TIME HERE
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
