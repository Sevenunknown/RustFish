use macroquad::prelude::*;
use macroquad::ui;
use chess::{Board, ChessMove, Color as ChessColor, Piece, Square, Game, MoveGen, GameResult};
use std::fmt;
use ::rand::Rng;

mod engine1;
use engine1::Engine;

#[derive(Clone, Copy)]
struct Highlight {
    square: Square,
    color: macroquad::color::Color,
}

fn get_opposite(colour: chess::Color) -> chess::Color {
    match colour {
        chess::Color::White => chess::Color::Black,
        chess::Color::Black => chess::Color::White
    }
}
const DEPTH: i32 = 6;
#[derive(PartialEq, Eq)]
enum GameMode {
    Menu,
    HumanVsAI { human_color: ChessColor },
    AIvsAI,
}

fn reset_ai() -> Engine {
    Engine::new(chess::Color::White, DEPTH)
}


#[derive(Debug, Clone)]
struct MoveRecord {
    move_number: usize,
    white_move: Option<String>,
    black_move: Option<String>,
    san: String, // Standard Algebraic Notation
    from: Square,
    to: Square,
    piece: Piece,
    captured: Option<Piece>,
    promotion: Option<Piece>,
}

impl fmt::Display for MoveRecord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.san)
    }
}

fn move_to_san(board: &Board, mv: ChessMove) -> String {
    let mut san = String::new();
    let piece = board.piece_on(mv.get_source()).unwrap();
    let colour = board.color_on(mv.get_source()).unwrap();
    let is_capture = board.piece_on(mv.get_dest()).is_some();
    let is_pawn = piece == Piece::Pawn;
    
    // Handle pawn moves
    if is_pawn {
        if is_capture {
            san.push_str(&format!("{}", mv.get_source().get_file().to_index().to_string().to_lowercase()));
        }
    } else {
        // Piece moves
        san.push_str(&piece.to_string(colour).to_uppercase());
        
        // Handle disambiguation
        let mut needs_file = false;
        let mut needs_rank = false;
        
        for other_move in MoveGen::new_legal(board) {
            if other_move.get_dest() == mv.get_dest() && 
                other_move.get_source() != mv.get_source() &&
                board.piece_on(other_move.get_source()) == Some(piece) {
                if mv.get_source().get_file() != other_move.get_source().get_file() {
                    needs_file = true;
                } else {
                    needs_rank = true;
                }
            }
        }
        
        if needs_file || needs_rank {
            if needs_file {
                san.push_str(&mv.get_source().get_file().to_index().to_string().to_lowercase());
            }
            if needs_rank {
                san.push_str(&mv.get_source().get_rank().to_index().to_string());
            }
        }
    }
    
    // Capture
    if is_capture {
        if is_pawn && !san.contains(mv.get_source().get_file().to_index().to_string().as_str()) {
            san.push_str(&mv.get_source().get_file().to_index().to_string().to_lowercase());
        }
        san.push('x');
    }
    
    // Destination
    san.push_str(&mv.get_dest().to_string());
    
    // Promotion
    if let Some(prom) = mv.get_promotion() {
        san.push('=');
        san.push_str(&prom.to_string(colour).to_uppercase());
    }
    
    // Check/checkmate
    let mut new_board = board.make_move_new(mv);
    if new_board.checkers().popcnt() > 0 {
        if MoveGen::new_legal(&new_board).count() == 0 {
            san.push('#');
        } else {
            san.push('+');
        }
    }
    
    san
}

fn record_move(game: &Game, mv: ChessMove, move_history: &mut Vec<MoveRecord>) {
    let board = game.current_position();
    let is_white = game.side_to_move() == ChessColor::White;
    
    let record = MoveRecord {
        move_number: if is_white {
            move_history.len() + 1
        } else {
            move_history.len()
        },
        white_move: if is_white { Some(move_to_san(&board, mv)) } else { None },
        black_move: if !is_white { Some(move_to_san(&board, mv)) } else { None },
        san: move_to_san(&board, mv),
        from: mv.get_source(),
        to: mv.get_dest(),
        piece: board.piece_on(mv.get_source()).unwrap(),
        captured: board.piece_on(mv.get_dest()),
        promotion: mv.get_promotion(),
    };
    
    if is_white {
        move_history.push(record);
    } else if let Some(last) = move_history.last_mut() {
        last.black_move = record.black_move.clone();
    } else {
        // This handles the case where black moves first (unusual but possible)
        move_history.push(MoveRecord {
            move_number: 1,
            white_move: None,
            black_move: record.black_move.clone(),
            ..record
        });
    }
}
fn export_to_pgn(move_history: &[MoveRecord], result: GameResult) -> String {
    let mut pgn = String::new();
    
    // PGN headers
    pgn.push_str("[Event \"Computer Game\"]\n");
    pgn.push_str("[Site \"?\"]\n");
    pgn.push_str("[Date \"????.??.??\"]\n");
    pgn.push_str("[Round \"-\"]\n");
    pgn.push_str("[White \"AI\"]\n");
    pgn.push_str("[Black \"AI\"]\n");
    pgn.push_str(&format!("[Result \"{}\"]\n", match result {
        GameResult::WhiteCheckmates => "1-0",
        GameResult::BlackCheckmates => "0-1",
        GameResult::Stalemate => "1/2-1/2",
        GameResult::DrawAccepted => "1/2-1/2",
        GameResult::DrawDeclared => "1/2-1/2",
        _ => "*",
    }));
    pgn.push_str("\n");
    
    // Moves
    for (i, record) in move_history.iter().enumerate() {
        if i % 2 == 0 {
            pgn.push_str(&format!("{}. ", (i / 2) + 1));
        }
        pgn.push_str(&record.san);
        pgn.push(' ');
    }
    
    // Result
    pgn.push_str(match result {
        GameResult::WhiteCheckmates => "1-0",
        GameResult::BlackCheckmates => "0-1",
        GameResult::Stalemate => "1/2-1/2",
        GameResult::DrawAccepted => "1/2-1/2",
        GameResult::DrawDeclared => "1/2-1/2",
        _ => "*",
    });
    
    pgn
}


#[macroquad::main("Chess")]
async fn main() {
    let sprite_sheet = load_texture("ChessPiecesArray.png").await.unwrap();
    sprite_sheet.set_filter(FilterMode::Nearest);
    let mut move_history: Vec<MoveRecord> = Vec::new();
    let mut game_mode = GameMode::Menu;
    let mut game = Game::new();
    let mut selected_square: Option<Square> = None;
    let mut highlights: Vec<Highlight> = vec![];

    let mut ai_moved_this_turn = false;
    let mut last_side_to_move: Option<ChessColor> = None;
    let mut frame_delay = 0;
    let mut last_eval: f32 = 0.0;
    let mut last_eval_side: Option<ChessColor> = None;

    let mut AI1 = Engine::new(chess::Color::White, DEPTH);
    let mut AI2 = Engine::new(chess::Color::Black, DEPTH);
    let mut AI = Engine::new(chess::Color::Black, DEPTH);
    let AI_turn = 1;

    loop {
        if frame_delay > 0 {
            frame_delay -= 1;
        }
        if is_key_pressed(KeyCode::P) {
            let r: chess::GameResult;
            if game.result().is_none() {
                r = chess::GameResult::DrawAccepted
            }
            else {
                r = game.result().unwrap()
            }
            let png = export_to_pgn(&move_history, r);
            println!("{:?}", png);
        }
        match game_mode {
            GameMode::Menu => {
                draw_text("Chess Game Modes:", 50.0, 150.0, 40.0, BLACK);
                draw_text("Press 1: Human vs AI", 50.0, 220.0, 30.0, DARKGRAY);
                draw_text("Press 2: AI vs AI", 50.0, 270.0, 30.0, DARKGRAY);
                draw_text("Press R: Reset (Restart Menu)", 50.0, 320.0, 25.0, GRAY);

                if is_key_pressed(KeyCode::Key1) {
                    let human_white = ::rand::thread_rng().gen_bool(0.5);
                    let human_color = if human_white { ChessColor::White } else { ChessColor::Black };
                    game = Game::new();
                    selected_square = None;
                    highlights.clear();
                    ai_moved_this_turn = false;
                    last_side_to_move = None;
                    frame_delay = 2;
                    last_eval = 0.0;
                    last_eval_side = None;
                    move_history.clear();
                    AI = reset_ai();
                    AI.set_colour(get_opposite(human_color));
                    game_mode = GameMode::HumanVsAI { human_color };
                }
                if is_key_pressed(KeyCode::Key2) {
                    game = Game::new();
                    selected_square = None;
                    highlights.clear();
                    ai_moved_this_turn = false;
                    last_side_to_move = None;
                    frame_delay = 2;
                    move_history.clear();
                    AI1 = reset_ai();
                    AI2 = reset_ai();
                    AI1.set_colour(chess::Color::White);
                    AI2.set_colour(chess::Color::Black);
                    last_eval = 0.0;
                    last_eval_side = None;
                    game_mode = GameMode::AIvsAI;
                }
            }

            GameMode::HumanVsAI { human_color } => {
                draw_board();
                draw_pieces(&game.current_position(), &sprite_sheet);
                draw_highlights(&highlights);

                if is_key_pressed(KeyCode::R) {
                    if let Some(result) = game.result() {
                        println!("Game PGN:\n{}", export_to_pgn(&move_history, result));
                    }
                    game_mode = GameMode::Menu;
                    move_history.clear();
                    continue;
                }

                let color_to_move = game.side_to_move();

                if color_to_move == human_color && game.result().is_none() {
                    if is_mouse_button_pressed(MouseButton::Left) {
                        let (mx, my) = mouse_position();
                        if let Some(sq) = mouse_pos_to_square(mx, my) {
                            let board = game.current_position();
                            if let Some(selected) = selected_square {
                                let promotion_rank = match board.color_on(selected) {
                                    Some(ChessColor::White) => 7,
                                    Some(ChessColor::Black) => 0,
                                    _ => 8,
                                };
                                let is_pawn = board.piece_on(selected) == Some(Piece::Pawn);
                                let dest_rank = sq.get_rank().to_index();
                                let mv = if is_pawn && dest_rank == promotion_rank {
                                    ChessMove::new(selected, sq, Some(Piece::Queen))
                                } else {
                                    ChessMove::new(selected, sq, None)
                                };
                                if board.legal(mv) {
                                    record_move(&game, mv, &mut move_history);
                                    game.make_move(mv);
                                    selected_square = None;
                                    highlights.clear();
                                    ai_moved_this_turn = false;
                                    frame_delay = 2;
                                } else if board.piece_on(sq) != None && board.color_on(sq) == Some(color_to_move) {
                                    selected_square = Some(sq);
                                    highlights = generate_highlights(&board, sq);
                                } else {
                                    selected_square = None;
                                    highlights.clear();
                                }
                            } else if let Some(_p) = board.piece_on(sq) {
                                if board.color_on(sq) == Some(color_to_move) {
                                    selected_square = Some(sq);
                                    highlights = generate_highlights(&board, sq);
                                }
                            }
                        }
                    }
                } else if color_to_move != human_color && game.result().is_none() && !ai_moved_this_turn && frame_delay == 0 {
                    if let mv = AI.get_move(&game.current_position()).unwrap() {
                        println!("AI Played {mv}");
                        record_move(&game, mv, &mut move_history);
                        game.make_move(mv);
                        selected_square = None;
                        highlights.clear();
                        ai_moved_this_turn = true;
                        frame_delay = 2;
                        last_eval = AI.get_current_eval();
                        last_eval_side = Some(color_to_move);
                    }
                }

                if let Some(result) = game.result() {
                    draw_text(
                        &format!("Game over: {:?}", result),
                        10.0,
                        screen_height() - 40.0,
                        30.0,
                        RED,
                    );
                    let pgn = export_to_pgn(&move_history, result);
                    println!("PGN:\n{}", pgn);
                }

                if let eval = last_eval {
                    let label = format!("AI eval: {:.2}", eval);
                    draw_text(&label, screen_width()-screen_width()*0.2, 30.0, 25.0, DARKBLUE);
                }
            }

            GameMode::AIvsAI => {
                clear_background(WHITE);
                draw_board();
                draw_pieces(&game.current_position(), &sprite_sheet);

                if is_key_pressed(KeyCode::R) {
                    if let Some(result) = game.result() {
                        println!("Game PGN:\n{}", export_to_pgn(&move_history, result));
                    }
                    game_mode = GameMode::Menu;
                    move_history.clear();
                    continue;
                }

                let current_side = game.side_to_move();

                if Some(current_side) != last_side_to_move {
                    last_side_to_move = Some(current_side);
                    ai_moved_this_turn = false;
                }

                if game.result().is_none() {
                    if !ai_moved_this_turn && frame_delay == 0 {
                        if last_side_to_move == Some(chess::Color::White) {
                            let mv = AI1.get_move(&game.current_position()).unwrap();
                            record_move(&game, mv, &mut move_history);
                            game.make_move(mv);
                            ai_moved_this_turn = true;
                            frame_delay = 2;
                            last_eval = AI1.get_current_eval();
                            last_eval_side = Some(current_side);
                        } else {
                            let mv = AI2.get_move(&game.current_position()).unwrap();
                            record_move(&game, mv, &mut move_history);
                            game.make_move(mv);
                            ai_moved_this_turn = true;
                            frame_delay = 2;
                            last_eval = AI2.get_current_eval();
                            last_eval_side = Some(current_side);
                        }
                    }
                } else if let Some(result) = game.result() {
                    draw_text(
                        &format!("Game over: {:?}", result),
                        10.0,
                        screen_height() - 40.0,
                        30.0,
                        RED,
                    );
                }

                let label1 = format!("AI 1 Eval: {:.2}", AI1.get_current_eval());
                let label2 = format!("AI 2 Eval: {:.2}", AI2.get_current_eval());
                draw_text(&label1, screen_width()-screen_width()*0.2, screen_height()-screen_height()*0.2, 25.0, DARKBLUE);
                draw_text(&label2, screen_width()-screen_width()*0.2, screen_height()-screen_height()*0.8, 25.0, DARKBLUE);
            }
        }

        if frame_delay > 0 {
            draw_text("Thinking...", screen_width()-screen_width()*0.2, screen_height() - 10.0, 20.0, DARKBLUE);
        }

        if frame_delay == -1 {
            let mut c = 0;
            println!("Current Moves");
            for mv in move_history.clone() {
                if c % 2 == 0 {
                    println!("White: {mv}")
                } else {
                    println!("Black: {mv}")
                }
                c += 1;
            }
        }

        next_frame().await;
    }
}

fn draw_board() {
    let tile_size = screen_width().min(screen_height()) / 8.0;
    for y in 0..8 {
        for x in 0..8 {
            let is_light = (x + y) % 2 == 0;
            let color = if is_light { LIGHTGRAY } else { DARKGRAY };
            draw_rectangle(x as f32 * tile_size, y as f32 * tile_size, tile_size, tile_size, color);
        }
    }
}

fn draw_pieces(board: &Board, sheet: &Texture2D) {
    let tile_size = screen_width().min(screen_height()) / 8.0;
    let sprite_width = sheet.width() / 6.0;
    let sprite_height = sheet.height() / 2.0;

    for sq_index in 0..64 {
        let sq = unsafe { Square::new(sq_index) };
        if let Some(piece) = board.piece_on(sq) {
            let color = board.color_on(sq).unwrap();
            let (x, y) = (sq.get_file().to_index(), 7 - sq.get_rank().to_index());

            let kind_index = match piece {
                Piece::Queen => 0,
                Piece::King => 1,
                Piece::Rook => 2,
                Piece::Knight => 3,
                Piece::Bishop => 4,
                Piece::Pawn => 5,
            };
            let row = match color {
                ChessColor::Black => 0,
                ChessColor::White => 1,
            };

            let source_rect = Rect::new(
                kind_index as f32 * sprite_width,
                row as f32 * sprite_height,
                sprite_width,
                sprite_height,
            );

            draw_texture_ex(
                sheet,
                x as f32 * tile_size,
                y as f32 * tile_size,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(Vec2::new(tile_size, tile_size)),
                    source: Some(source_rect),
                    ..Default::default()
                },
            );
        }
    }
}

fn draw_highlights(highlights: &[Highlight]) {
    let tile_size = screen_width().min(screen_height()) / 8.0;
    for h in highlights {
        let x = h.square.get_file().to_index();
        let y = 7 - h.square.get_rank().to_index();
        let mut color = h.color;
        color.a = 0.5;
        draw_rectangle(
            x as f32 * tile_size,
            y as f32 * tile_size,
            tile_size,
            tile_size,
            color,
        );
    }
}

fn mouse_pos_to_square(x: f32, y: f32) -> Option<Square> {
    let tile_size = screen_width().min(screen_height()) / 8.0;
    let file = (x / tile_size).floor() as usize;
    let rank = 7 - (y / tile_size).floor() as usize;
    if file < 8 && rank < 8 {
        Some(Square::make_square(
            chess::Rank::from_index(rank),
            chess::File::from_index(file),
        ))
    } else {
        None
    }
}

fn generate_highlights(board: &Board, sq: Square) -> Vec<Highlight> {
    let mut result = vec![Highlight {
        square: sq,
        color: YELLOW,
    }];
    let movegen = MoveGen::new_legal(board);
    for mv in movegen {
        if mv.get_source() == sq {
            result.push(Highlight {
                square: mv.get_dest(),
                color: GREEN,
            });
        }
    }
    result
}
