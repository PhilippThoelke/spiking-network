use nannou::prelude::*;

pub fn to_screen_coords(pos: Vec2, win: Rect) -> Vec2 {
    vec2(
        pos.x * win.w() / crate::ASPECT_RATIO - 0.5 * win.w(),
        pos.y * win.h() - 0.5 * win.h(),
    )
}
