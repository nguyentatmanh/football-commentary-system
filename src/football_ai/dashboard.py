import os
import json
import glob
import subprocess
import streamlit as st
import pandas as pd

def get_web_compatible_video(video_path):
    if not video_path:
        return None
    
    dir_name = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    h264_name = base_name.replace(".mp4", "_h264.mp4")
    h264_path = os.path.join(dir_name, h264_name)
    
    if os.path.exists(h264_path):
        return h264_path
        
    with st.spinner("Đang chuyển đổi định dạng video sang H.264 để phát trên trình duyệt... (Chỉ chạy lần đầu)"):
        try:
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                h264_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(h264_path):
                return h264_path
            else:
                st.error(f"Lỗi khi chuyển đổi video bằng FFmpeg: {result.stderr}")
        except Exception as e:
            st.error(f"Không thể gọi FFmpeg để chuyển đổi video (Hãy chắc chắn FFmpeg đã cài trong PATH): {e}")
            
    return video_path

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Football AI Analytics Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS để giao diện trông hiện đại và chuyên nghiệp
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e222b;
        border-radius: 4px 4px 0px 0px;
        color: #a0a0a5;
        font-weight: 600;
        font-size: 15px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00dffc !important;
        color: #0e1117 !important;
    }
    .metric-card {
        background-color: #1f2430;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #00dffc;
        margin-bottom: 15px;
    }
    .legend-box {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-size: 14px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Tiêu đề chính
st.markdown("""
    <div style="background-color: #1e222b; padding: 24px; border-radius: 8px; margin-bottom: 25px; border-bottom: 4px solid #00dffc;">
        <h1 style="color: #00dffc; margin: 0; font-family: 'Inter', sans-serif;">🏆 Football AI Broadcast Dashboard</h1>
        <p style="color: #a0a0a5; margin: 8px 0 0 0; font-size: 16px;">
            Hệ thống phân tích video bóng đá tự động: Phát hiện đối tượng, Theo dõi quỹ đạo, Phân loại đội bóng và Tạo bản đồ nhiệt.
        </p>
    </div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CẤU HÌNH -----------------
st.sidebar.markdown("### 🛠️ Cấu hình dữ liệu")

# Thư mục gốc chứa các kết quả
DEFAULT_OUTPUTS_DIR = "data/outputs"
if not os.path.exists(DEFAULT_OUTPUTS_DIR):
    st.sidebar.warning(f"Không tìm thấy thư mục mặc định `{DEFAULT_OUTPUTS_DIR}`. Hãy chọn thư mục thủ công.")
    outputs_dir = st.sidebar.text_input("Đường dẫn thư mục kết quả:", value="data")
else:
    outputs_dir = DEFAULT_OUTPUTS_DIR

# Quét các thư mục con trong data/outputs để lấy danh sách các lần phân tích (run)
run_folders = []
if os.path.exists(outputs_dir):
    run_folders = [f for f in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, f))]
    run_folders = [f for f in run_folders if f not in ["classification", "detection", "tracking", "demo"]]

if run_folders:
    selected_run = st.sidebar.selectbox("Chọn lượt phân tích (Run ID):", sorted(run_folders))
    selected_path = os.path.join(outputs_dir, selected_run)
else:
    st.sidebar.info("Không phát hiện thư mục phân tích cụ thể. Đang đọc trực tiếp từ thư mục kết quả.")
    selected_path = outputs_dir

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Thông tin giải thích lớp vẽ (Layers)")
st.sidebar.markdown("""
- 🟢 **Phát hiện (Detection):** Vẽ bounding box bao quanh Cầu thủ, Trọng tài và Bóng.
- 🆔 **Theo dõi (Tracking):** Đánh số ID cố định trên mỗi đối tượng để theo dõi hành trình di chuyển qua các khung hình.
- 🎨 **Phân loại (Classification):** 
  - **Màu Hồng (Pink):** Đội A (Team A)
  - **Màu Xanh Nhạt (Azure):** Đội B (Team B)
  - **Màu Xanh Lá (Green):** Trọng tài
  - **Màu Vàng (Yellow):** Quả Bóng
- 🗺️ **Bản đồ Radar:** Góc nhìn 2D từ trên xuống (Top-down view) để quan sát sơ đồ đội hình chiến thuật ở góc màn hình.
""")

# Kiểm tra sự tồn tại của thư mục kết quả
if not os.path.exists(selected_path):
    st.error(f"Thư mục kết quả `{selected_path}` không tồn tại. Vui lòng kiểm tra lại cấu hình.")
else:
    # ----------------- ĐỌC DỮ LIỆU KẾT QUẢ -----------------
    
    # 1. Tìm tất cả các file video .mp4 đệ quy trong thư mục kết quả
    all_mp4_files = glob.glob(os.path.join(selected_path, "**", "*.mp4"), recursive=True)
    all_mp4_files = [f for f in all_mp4_files if not f.endswith("_h264.mp4")]
    
    available_videos = {}
    for f in all_mp4_files:
        rel_path = os.path.relpath(f, selected_path)
        # Tạo tên thân thiện cho selectbox
        if rel_path == "analytics_video.mp4" or rel_path == "analytics_video_with_audio.mp4":
            name = "Tổng hợp (Full Pipeline)"
            # Ưu tiên bản video có tiếng bình luận (audio) nếu có cả 2
            if "with_audio" in rel_path:
                available_videos[name] = f
            elif name not in available_videos:
                available_videos[name] = f
        elif "tracking" in rel_path.lower():
            available_videos["Theo dõi đối tượng (Tracking)"] = f
        elif "classification" in rel_path.lower():
            available_videos["Phân loại đội bóng (Classification)"] = f
        else:
            available_videos[f"Mô-đun khác: {rel_path}"] = f
        
    # 2. Đọc possession_summary.json
    possession_data = None
    possession_path = os.path.join(selected_path, "possession_summary.json")
    if os.path.exists(possession_path):
        with open(possession_path, 'r', encoding='utf-8') as f:
            possession_data = json.load(f)

    # 3. Đọc movement_summary.json
    movement_data = None
    movement_path = os.path.join(selected_path, "movement_summary.json")
    if os.path.exists(movement_path):
        with open(movement_path, 'r', encoding='utf-8') as f:
            movement_data = json.load(f)

    # 4. Tìm các ảnh heatmap
    heatmap_dir = os.path.join(selected_path, "heatmaps")
    heatmap_files = {}
    if os.path.exists(heatmap_dir):
        images = glob.glob(os.path.join(heatmap_dir, "*.png"))
        for img in images:
            basename = os.path.basename(img)
            name = basename.replace(".png", "").replace("_heatmap", "").replace("_", " ").title()
            heatmap_files[name] = img

    # ----------------- THIẾT LẬP CÁC TABS GIAO DIỆN -----------------
    tab_video, tab_heatmaps, tab_possession, tab_movement = st.tabs([
        "📺 Video Phân Tích", 
        "🗺️ Bản Đồ Nhiệt (Heatmaps)", 
        "📊 Kiểm Soát Bóng", 
        "🏃 Thống Kê Di Chuyển"
    ])

    # === TAB 1: VIDEO PHÂN TÍCH ===
    with tab_video:
        st.subheader("📺 Kết quả video sau phân tích")
        
        if available_videos:
            # Hộp chọn video của từng mô-đun
            selected_video_name = st.selectbox(
                "Chọn video hiển thị theo mô-đun phân tích:",
                options=list(available_videos.keys()),
                index=0
            )
            video_file = available_videos[selected_video_name]
            
            # Chuyển đổi sang H.264 tương thích trình duyệt
            video_file_compatible = get_web_compatible_video(video_file)
            
            if video_file_compatible:
                st.info(f"Đang phát: `{selected_video_name}` ({os.path.basename(video_file)})")
                st.video(video_file_compatible)
                
            # Ghi chú chú giải các lớp trên video
            st.markdown("### Chú thích các thành phần vẽ đè (Overlays)")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("""
                <div class="legend-box" style="background-color: rgba(255, 20, 147, 0.2); border: 2px solid rgb(255, 20, 147);">
                    🩷 Đội A (Team A - Pink)
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="legend-box" style="background-color: rgba(0, 191, 255, 0.2); border: 2px solid rgb(0, 191, 255);">
                    🩵 Đội B (Team B - Azure)
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                <div class="legend-box" style="background-color: rgba(0, 255, 0, 0.2); border: 2px solid rgb(0, 255, 0);">
                    🟢 Trọng Tài (Referee - Green)
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown("""
                <div class="legend-box" style="background-color: rgba(255, 215, 0, 0.2); border: 2px solid rgb(255, 215, 0);">
                    💛 Quả Bóng (Ball - Yellow)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Không tìm thấy tệp video đầu ra (.mp4) trong thư mục này.")

    # === TAB 2: BẢN ĐỒ NHIỆT (HEATMAPS) ===
    with tab_heatmaps:
        st.subheader("🗺️ Bản đồ nhiệt mật độ vị trí (Heatmaps)")
        st.write("Bản đồ nhiệt thể hiện mật độ xuất hiện và di chuyển của các đối tượng trên sân bóng (được chuẩn hóa từ phép biến đổi homography 2D).")
        
        if heatmap_files:
            # Chọn loại heatmap để hiển thị
            selected_heatmap = st.selectbox(
                "Chọn bản đồ nhiệt cần hiển thị:", 
                options=list(heatmap_files.keys()),
                index=0
            )
            
            # Hiển thị ảnh heatmap được chọn
            img_path = heatmap_files[selected_heatmap]
            st.image(img_path, caption=f"Bản đồ nhiệt: {selected_heatmap}", use_container_width=True)
            
            # Hiển thị lưới so sánh song song nếu có đủ ảnh của Đội A và Đội B
            st.markdown("---")
            st.subheader("⚖️ So sánh Đội A vs Đội B")
            col_a, col_b = st.columns(2)
            
            team_a_img = heatmap_files.get("Team A")
            team_b_img = heatmap_files.get("Team B")
            
            with col_a:
                if team_a_img:
                    st.image(team_a_img, caption="Bản đồ nhiệt Đội A", use_container_width=True)
                else:
                    st.info("Không tìm thấy bản đồ nhiệt Đội A.")
            with col_b:
                if team_b_img:
                    st.image(team_b_img, caption="Bản đồ nhiệt Đội B", use_container_width=True)
                else:
                    st.info("Không tìm thấy bản đồ nhiệt Đội B.")
        else:
            st.warning("Không tìm thấy các tệp bản đồ nhiệt (.png) trong thư mục `heatmaps/`.")

    # === TAB 3: KIỂM SOÁT BÓNG (POSSESSION) ===
    with tab_possession:
        st.subheader("📊 Thống kê tỷ lệ kiểm soát bóng (Possession)")
        
        if possession_data:
            # Lấy thông số
            total_frames = possession_data.get("evaluated_frames", 0)
            dist = possession_data.get("possession_distribution", {})
            split = possession_data.get("broadcast_split", {})
            
            # Chuyển đổi sang dạng phần trăm
            team_a_pct = dist.get("team_a_percent", 0.0)
            team_b_pct = dist.get("team_b_percent", 0.0)
            none_pct = dist.get("none_percent", 0.0)
            
            a_ratio = split.get("team_a_ratio", 0.0)
            b_ratio = split.get("team_b_ratio", 0.0)
            
            # Khối chỉ số chính (Metrics)
            col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
            with col_metric_1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #a0a0a5;">Kiểm soát Đội A</h4>
                    <h2 style="margin: 10px 0 0 0; color: rgb(255, 20, 147);">{team_a_pct}%</h2>
                    <p style="margin: 5px 0 0 0; font-size: 13px; color: #707075;">Tổng thời lượng trận đấu</p>
                </div>
                """, unsafe_allow_html=True)
            with col_metric_2:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #00bfff;">
                    <h4 style="margin: 0; color: #a0a0a5;">Kiểm soát Đội B</h4>
                    <h2 style="margin: 10px 0 0 0; color: #00bfff;">{team_b_pct}%</h2>
                    <p style="margin: 5px 0 0 0; font-size: 13px; color: #707075;">Tổng thời lượng trận đấu</p>
                </div>
                """, unsafe_allow_html=True)
            with col_metric_3:
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: #707075;">
                    <h4 style="margin: 0; color: #a0a0a5;">Bóng tranh chấp / Trôi tự do</h4>
                    <h2 style="margin: 10px 0 0 0; color: #ffffff;">{none_pct}%</h2>
                    <p style="margin: 5px 0 0 0; font-size: 13px; color: #707075;">Không bên nào giữ bóng</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Thanh hiển thị tỷ lệ sở hữu thực tế (Loại bỏ thời gian bóng chết/trôi tự do)
            st.markdown("---")
            st.subheader("📺 Tỷ lệ hiển thị trên truyền hình (Broadcast Split)")
            st.write("Chỉ tính khoảng thời gian bóng thực sự nằm trong tầm kiểm soát của một trong hai đội:")
            
            # Vẽ thanh phần trăm sở hữu
            col_bar_a, col_bar_b = st.columns([int(a_ratio) if a_ratio > 0 else 1, int(b_ratio) if b_ratio > 0 else 1])
            with col_bar_a:
                st.markdown(f"""
                <div style="background-color: rgb(255, 20, 147); text-align: center; padding: 10px; border-radius: 4px 0 0 4px; font-weight: bold; color: white;">
                    Đội A: {a_ratio:.1f}%
                </div>
                """, unsafe_allow_html=True)
            with col_bar_b:
                st.markdown(f"""
                <div style="background-color: rgb(0, 191, 255); text-align: center; padding: 10px; border-radius: 0 4px 4px 0; font-weight: bold; color: white;">
                    Đội B: {b_ratio:.1f}%
                </div>
                """, unsafe_allow_html=True)
                
            # Biểu đồ cột phân phối sở hữu
            st.markdown("### Biểu đồ phân phối sở hữu (%)")
            chart_data = pd.DataFrame({
                "Đội bóng": ["Đội A (Hồng)", "Đội B (Xanh)", "Tranh chấp/Tự do"],
                "Phần trăm sở hữu": [team_a_pct, team_b_pct, none_pct]
            })
            st.bar_chart(chart_data.set_index("Đội bóng"))
            
        else:
            st.warning("Không tìm thấy tệp `possession_summary.json` trong thư mục này.")

    # === TAB 4: THỐNG KÊ DI CHUYỂN ===
    with tab_movement:
        st.subheader("🏃 Thống kê di chuyển và tốc độ (Movement Analytics)")
        
        if movement_data:
            # Chuyển đổi dữ liệu JSON sang DataFrame
            df = pd.DataFrame(movement_data)
            
            # Map thông tin để dễ đọc hơn
            role_map = {"player": "Cầu thủ", "goalkeeper": "Thủ môn", "referee": "Trọng tài", "ball": "Bóng"}
            df["vai_tro"] = df["role"].map(role_map).fillna(df["role"])
            
            def map_team(row):
                if row["role"] == "ball":
                    return "Bóng"
                elif row["role"] == "referee":
                    return "Trọng tài"
                elif row["team_id"] == 0:
                    return "Đội A (Hồng)"
                elif row["team_id"] == 1:
                    return "Đội B (Xanh)"
                return "Không xác định"
                
            df["doi_bong"] = df.apply(map_team, axis=1)
            
            # Định dạng lại các cột số để hiển thị đẹp hơn
            df["Quãng đường (m)"] = df["total_distance_m"].round(2)
            df["Tốc độ TB (km/h)"] = (df["average_speed_mps"] * 3.6).round(2)
            df["Tốc độ cực đại (km/h)"] = (df["max_speed_mps"] * 3.6).round(2)
            df["ID Đối tượng"] = df["track_id"]
            
            # Lọc bỏ bóng để vẽ biểu đồ so sánh cầu thủ
            df_players = df[df["role"].isin(["player", "goalkeeper"])]
            
            # Tạo các bộ lọc trên giao diện
            col_filter_1, col_filter_2 = st.columns(2)
            with col_filter_1:
                selected_team_filter = st.selectbox(
                    "Lọc theo Đội bóng:",
                    options=["Tất cả", "Đội A (Hồng)", "Đội B (Xanh)"]
                )
            with col_filter_2:
                sort_by = st.selectbox(
                    "Sắp xếp theo:",
                    options=["Quãng đường (m)", "Tốc độ cực đại (km/h)", "Tốc độ TB (km/h)"],
                    index=0
                )
                
            # Áp dụng bộ lọc
            df_filtered = df_players.copy()
            if selected_team_filter != "Tất cả":
                df_filtered = df_filtered[df_filtered["doi_bong"] == selected_team_filter]
                
            df_filtered = df_filtered.sort_values(by=sort_by, ascending=False)
            
            # Hiển thị bảng số liệu cầu thủ
            st.markdown("### Bảng xếp hạng chỉ số vật lý cầu thủ")
            display_cols = ["ID Đối tượng", "doi_bong", "vai_tro", "Quãng đường (m)", "Tốc độ TB (km/h)", "Tốc độ cực đại (km/h)"]
            st.dataframe(df_filtered[display_cols].reset_index(drop=True), use_container_width=True)
            
            # Biểu đồ so sánh
            st.markdown("---")
            st.subheader(f"📊 So sánh {sort_by} giữa các cầu thủ")
            
            chart_df = df_filtered.copy()
            chart_df["Label"] = "Cầu thủ " + chart_df["ID Đối tượng"].astype(str) + " (" + chart_df["doi_bong"] + ")"
            
            # Vẽ biểu đồ cột
            st.bar_chart(chart_df.set_index("Label")[sort_by])
            
            # Thống kê nhanh trọng tài và bóng
            st.markdown("---")
            st.subheader("👤 Trọng tài & Quả bóng")
            df_ref_ball = df[df["role"].isin(["referee", "ball"])]
            st.table(df_ref_ball[["vai_tro", "Quãng đường (m)", "Tốc độ TB (km/h)", "Tốc độ cực đại (km/h)"]].reset_index(drop=True))
            
        else:
            st.warning("Không tìm thấy tệp `movement_summary.json` trong thư mục này.")
