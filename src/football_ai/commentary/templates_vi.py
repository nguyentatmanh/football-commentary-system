"""
Vietnamese commentary templates for various football events.
"""

TEMPLATES = {
    "possession_change": [
        "Đội {team} đã giành lại quyền kiểm soát bóng.",
        "Một pha đoạt bóng tốt của đội {team}.",
        "Bóng đã thuộc về quyền kiểm soát của các cầu thủ đội {team}.",
        "Đội {team} đang có bóng triển khai."
    ],
    "loose_ball": [
        "Bóng đang ở trạng thái tranh chấp hỗn loạn.",
        "Chưa đội nào thực sự kiểm soát được trái bóng vào lúc này.",
        "Tình huống tranh chấp quyết liệt ở khu vực giữa sân.",
        "Bóng đang lăn tự do, chưa cầu thủ nào khống chế thành công."
    ],
    "deep_entry": [
        "Đội {team} đang đưa bóng vào khu vực nguy hiểm.",
        "Một pha lên bóng rất đáng chú ý của đội {team}.",
        "Đội {team} đã áp sát khu vực cấm địa của đối phương.",
        "Tình huống tổ chức tấn công sâu bên phần sân đối thủ của đội {team}."
    ],
    "high_speed_run": [
        "Cầu thủ số {player_id} đang tăng tốc rất nhanh bên phía đội nhà.",
        "Một pha bứt tốc cực kỳ mạnh mẽ của cầu thủ mang áo số {player_id}.",
        "Cầu thủ số {player_id} đang dùng tốc độ xé gió vượt qua đối thủ.",
        "Tình huống đi bóng tốc độ cao rất ấn tượng của cầu thủ {player_id}."
    ],
    "attack_build_up": [
        "Đội {team} đang kiên nhẫn tổ chức pha dàn xếp tấn công.",
        "Đội {team} đang triển khai bóng lên phía trước khá mạch lạc.",
        "Những đường chuyền liên tiếp nhằm kéo dãn đội hình đối phương của đội {team}.",
        "Đội {team} đang giữ cự ly đội hình và kiểm soát nhịp độ trận đấu tốt."
    ]
}

# Fallback templates in case dynamic fields are missing
FALLBACK_TEMPLATES = {
    "possession_change": "Một tình huống thay đổi quyền kiểm soát bóng trên sân.",
    "loose_ball": "Bóng đang trôi tự do, tranh chấp quyết liệt.",
    "deep_entry": "Bóng đã được đưa vào vòng cấm của đối thủ.",
    "high_speed_run": "Một pha bứt tốc ấn tượng vừa diễn ra.",
    "attack_build_up": "Nhịp độ trận đấu đang được đẩy nhanh với những pha triển khai bóng phối hợp."
}

def get_template(event_type: str, seed_id: int) -> str:
    """Returns a deterministic template based on seed ID to support test reproducibility."""
    options = TEMPLATES.get(event_type, [])
    if not options:
        return FALLBACK_TEMPLATES.get(event_type, "Tình huống diễn ra rất nhanh.")
    return options[seed_id % len(options)]
