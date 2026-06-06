import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as signal

def create_tracker():
    """建立追蹤器 (相容不同版本的 OpenCV)"""
    try:
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        return cv2.TrackerCSRT_create()

def analyze_roly_poly_physics(time_stamps, angles):
    """
    依據用戶觀察進行終極優化：
    1. 包容低阻尼系統 (短影片內無明顯衰減)
    2. 包容前期漂移 (以最後靜止點為基準)
    3. 加入雜訊過濾機制 (中值濾波)
    """
    if len(angles) < 10:
        print("數據過少，無法分析。")
        return None, None, None

    # =========================
    # 1. 加入「雜訊過濾機制」
    # =========================
    # 使用中值濾波器排除像素跳動產生的離群值 (防跳點)
    angles_filtered = signal.medfilt(angles, kernel_size=5)
    
    # 再進行平滑處理 (模擬慣性)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    angles_smooth = np.convolve(angles_filtered, kernel, mode='same')
    
    # 擷取最後 15% 的影格計算「最終靜止平衡點」
    tail_length = max(5, len(angles_smooth) // 7)
    final_resting_angle = np.mean(angles_smooth[-tail_length:]) 
    
    # 數據中心化 (圍繞最終歸宿震盪)
    centered = angles_smooth - final_resting_angle

    # =========================
    # 2. 物理特徵提取
    # =========================
    # 找出向右擺動 (正向) 與 向左擺動 (負向) 的所有局部波峰
    pos_peaks, _ = signal.find_peaks(centered, height=1.5)
    neg_peaks, _ = signal.find_peaks(-centered, height=1.5)

    # 綜合所有波峰位置，按時間排序
    all_peaks_idx = np.sort(np.concatenate((pos_peaks, neg_peaks)))
    
    # 【關鍵修正】：在這裡就先算出所有波峰的絕對振幅，供後面使用
    if len(all_peaks_idx) > 0:
        peak_amplitudes = np.abs(centered[all_peaks_idx])
    else:
        peak_amplitudes = np.array([])

    # =========================
    # 3. 加入「對稱交替擺動」判定
    # =========================
    has_phase_swing = False
    if len(all_peaks_idx) >= 3:
        # 檢查波峰的正負號序列是否為：+ - + - ...
        peak_centered_signs = np.sign(centered[all_peaks_idx])
        signs_diff = np.diff(peak_centered_signs)
        # 排除 0 (無符號變化) 或非負向變化 (-1-1=-2)
        has_phase_swing = np.all(signs_diff != 0) and (abs(np.mean(centered[all_peaks_idx])) < np.max(peak_amplitudes)*0.3)

    # =========================
    # 4. 加入「振幅包絡線」衰減分析
    # =========================
    is_decaying = False
    slope = 0
    if len(all_peaks_idx) >= 2:
        # 直接使用剛才已經算好的 peak_amplitudes
        slope, _ = np.polyfit(np.arange(len(peak_amplitudes)), peak_amplitudes, 1)
        # 包容低阻尼系統：斜率為負即可，不再強制小於 -0.1
        is_decaying = slope < -0.01

    # =========================
    # 5. 加入「週期穩定性」判定 (CV)
    # =========================
    is_period_stable = False
    period_cv = 1.0
    if len(all_peaks_idx) >= 3:
        peak_times = time_stamps[all_peaks_idx]
        periods = np.diff(peak_times) # 每半個週期的時間差
        period_cv = np.std(periods) / np.mean(periods) # 變異係數
        # 優於 0.15 即視為穩定週期 (真不倒翁固有特徵)
        is_period_stable = period_cv < 0.15

    # =========================
    # 6. 綜合判定與輸出報告
    # =========================
    print("\n===== 物理特徵分析報告 (終極防呆版) =====")
    print(f"最終靜止基準線 = {final_resting_angle:.2f}°")
    print(f"來回交替擺動 = {'通過' if has_phase_swing else '失敗'}")
    print(f"振幅衰減斜率 = {slope:.4f} (需 < -0.01 供主要通過)")
    print(f"週期穩定性 (CV) = {period_cv:.4f} (需 < 0.15 供備援通過)")

    # 只要符合 [來回擺動] 且 ([有能量衰減的主要條件] 或 [週期極度穩定的備援條件])
    if has_phase_swing and (is_decaying or is_period_stable):
        print("\n👉 最終判定：【真不倒翁】")
        if is_period_stable and not is_decaying:
            print("   (🎯 偵測到高度穩定的週期性擺動，推測為低阻尼/高品質不倒翁，故啟動週期穩定判定機制)")
    else:
        print("\n👉 最終判定：【非不倒翁】")
        print("   [未通過原因]")
        if not has_phase_swing:
            print("   - 未呈現圍繞中心點的正負對稱擺動 (疑似人為單一方向推動滾走)")
        if not is_decaying and not is_period_stable:
            print("   - 既無合理能量耗散，擺動週期也無規律性，非自然不倒翁運動")

    # 🚨 視覺化修復：回傳 angles_smooth 確保圖表座標系正確對齊原始影片角度
    return angles_smooth, final_resting_angle, all_peaks_idx


def main():
    video_path = "test_2.mp4"  # 請替換為你的影片路徑
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 '{video_path}'")
        return

    ret, frame = cap.read()
    if not ret:
        return

    # 選擇追蹤點
    print("請框選不倒翁的「頂部」特徵 (按 SPACE/ENTER 確認)")
    bbox_top = cv2.selectROI("Select TOP", frame, False)
    cv2.destroyWindow("Select TOP")

    print("請框選不倒翁的「底部」特徵 (按 SPACE/ENTER 確認)")
    bbox_bottom = cv2.selectROI("Select BOTTOM", frame, False)
    cv2.destroyWindow("Select BOTTOM")

    if bbox_top == (0,0,0,0) or bbox_bottom == (0,0,0,0):
        print("未完整選擇，結束程式。")
        cap.release()
        return

    tracker_top = create_tracker()
    tracker_bottom = create_tracker()
    tracker_top.init(frame, bbox_top)
    tracker_bottom.init(frame, bbox_bottom)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    angles = []
    time_stamps = []
    frame_id = 0

    print("開始追蹤，按 ESC 結束...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success_top, bbox_top = tracker_top.update(frame)
        success_bottom, bbox_bottom = tracker_bottom.update(frame)

        if success_top and success_bottom:
            # 取得中心點
            cx_top = bbox_top[0] + bbox_top[2]/2
            cy_top = bbox_top[1] + bbox_top[3]/2
            cx_bottom = bbox_bottom[0] + bbox_bottom[2]/2
            cy_bottom = bbox_bottom[1] + bbox_bottom[3]/2

            # 視覺化追蹤線
            cv2.circle(frame, (int(cx_top), int(cy_top)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(cx_bottom), int(cy_bottom)), 5, (255, 0, 0), -1)
            cv2.line(frame, (int(cx_bottom), int(cy_bottom)), (int(cx_top), int(cy_top)), (0, 255, 0), 2)

            # 計算夾角 (反正切函數)
            dx = cx_top - cx_bottom
            dy = cy_bottom - cy_top
            angle = math.degrees(math.atan2(dx, dy))
            
            angles.append(angle)
            time_stamps.append(frame_id / fps)

            cv2.putText(frame, f"Angle: {angle:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Physics Angle Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    angles = np.array(angles)
    time_stamps = np.array(time_stamps)

    # 執行物理特徵分析
    angles_smooth, angle_eq, peaks_idx = analyze_roly_poly_physics(time_stamps, angles)

    # 繪製專業數據圖表
    if angles_smooth is not None:
        plt.figure(figsize=(10, 5))
        
        # 原始雜訊資料
        plt.plot(time_stamps, angles, color='gray', alpha=0.3, label="Raw Angle (Noisy)")
        # 平滑過濾後的資料
        plt.plot(time_stamps, angles_smooth, color='blue', linewidth=2, label="Filtered Angle")
        # 最終靜止基準線
        plt.axhline(angle_eq, color='green', linestyle='--', label=f"Resting Eq ({angle_eq:.1f}°)")

        # 標記波峰 (因為 peaks_idx 是從 centered 找的，但兩者對齊，可直接對應)
        if len(peaks_idx) > 0:
            plt.scatter(time_stamps[peaks_idx], angles_smooth[peaks_idx], color='red', zorder=5, label="Detected Peaks")
            plt.plot(time_stamps[peaks_idx], angles_smooth[peaks_idx], color='red', linestyle=':', alpha=0.6)

        plt.title("Roly-Poly Motion Analysis: Oscillation & Decay")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Tilt Angle (Degrees)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()