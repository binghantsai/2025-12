import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as signal
import csv
import os

def create_tracker():
    """建立追蹤器 (相容不同版本的 OpenCV)"""
    try:
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        return cv2.TrackerCSRT_create()

def evaluate_tipping_risk(time_stamps, angles_smooth, angle_eq, critical_angle=45.0):
    """[預判系統] 動態穩定度評估與傾倒風險指數 (Risk Index) 計算"""
    theta_deg = angles_smooth - angle_eq
    theta_rad = np.radians(theta_deg)
    
    dt = np.mean(np.diff(time_stamps))
    if dt <= 0: dt = 1/30.0 

    omega = np.gradient(theta_rad, dt)
    alpha = np.gradient(omega, dt)
    kinetic_proxy = omega**2

    risk_index = np.zeros_like(theta_rad)
    theta_crit_rad = np.radians(critical_angle)

    for i in range(len(theta_rad)):
        angle_ratio = min(abs(theta_rad[i]) / theta_crit_rad, 1.0)
        risk_position = (angle_ratio ** 2) * 40  
        risk_kinetic = min(kinetic_proxy[i] / (3.0**2), 1.0) * 30  
        
        risk_accel = 0
        is_moving_away = (theta_rad[i] * omega[i]) > 0
        is_accelerating_outward = is_moving_away and ((omega[i] * alpha[i]) > 0)
        
        if is_moving_away: risk_accel += 10
        if is_accelerating_outward: risk_accel += min(abs(alpha[i]) / 10.0, 1.0) * 20 

        total_risk = risk_position + risk_kinetic + risk_accel
        risk_index[i] = min(max(total_risk, 0.0), 100.0)

    risk_index_smooth = signal.medfilt(risk_index, kernel_size=5)
    return risk_index_smooth, omega

def analyze_roly_poly_physics(time_stamps, angles):
    """依據用戶觀察進行終極優化，並結合 Risk Index 進行最終安全判定。"""
    if len(angles) < 10:
        print("數據過少，無法分析。")
        return None, None, None, None

    angles_filtered = signal.medfilt(angles, kernel_size=5)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    angles_smooth = np.convolve(angles_filtered, kernel, mode='same')
    
    tail_length = max(5, len(angles_smooth) // 7)
    final_resting_angle = np.mean(angles_smooth[-tail_length:]) 
    centered = angles_smooth - final_resting_angle

    pos_peaks, _ = signal.find_peaks(centered, height=1.5)
    neg_peaks, _ = signal.find_peaks(-centered, height=1.5)
    all_peaks_idx = np.sort(np.concatenate((pos_peaks, neg_peaks)))
    
    if len(all_peaks_idx) > 0:
        peak_amplitudes = np.abs(centered[all_peaks_idx])
    else:
        peak_amplitudes = np.array([])

    has_phase_swing = False
    if len(all_peaks_idx) >= 3:
        peak_centered_signs = np.sign(centered[all_peaks_idx])
        signs_diff = np.diff(peak_centered_signs)
        has_phase_swing = np.all(signs_diff != 0) and (abs(np.mean(centered[all_peaks_idx])) < np.max(peak_amplitudes)*0.3)

    is_decaying = False
    slope = 0
    if len(all_peaks_idx) >= 2:
        slope, _ = np.polyfit(np.arange(len(peak_amplitudes)), peak_amplitudes, 1)
        is_decaying = slope < -0.01

    is_period_stable = False
    period_cv = 1.0
    if len(all_peaks_idx) >= 3:
        peak_times = time_stamps[all_peaks_idx]
        periods = np.diff(peak_times)
        period_cv = np.std(periods) / np.mean(periods)
        is_period_stable = period_cv < 0.15

    risk_index_smooth, _ = evaluate_tipping_risk(time_stamps, angles_smooth, final_resting_angle, critical_angle=45.0)
    final_risk = np.mean(risk_index_smooth[-tail_length:])
    is_risk_safe = final_risk < 20.0

    print("\n===== 物理特徵分析報告 (終極防呆 + 動態預判版) =====")
    print(f"最終靜止基準線 = {final_resting_angle:.2f}°")
    print(f"來回交替擺動 = {'通過' if has_phase_swing else '失敗'}")
    print(f"振幅衰減斜率 = {slope:.4f} (需 < -0.01 供主要通過)")
    print(f"週期穩定性 (CV) = {period_cv:.4f} (需 < 0.15 供備援通過)")
    print(f"最終傾倒風險 = {final_risk:.1f}% (需 < 20.0% 以確認成功歸位)")

    if has_phase_swing and (is_decaying or is_period_stable) and is_risk_safe:
        print("\n👉 最終判定：【真不倒翁】")
    else:
        print("\n👉 最終判定：【非不倒翁 / 判定失敗】")

    return angles_smooth, final_resting_angle, all_peaks_idx, risk_index_smooth

def extract_ml_features(time_stamps, angles_smooth, peaks_idx, risk_index):
    """【新增】將物理波形轉換為 AI 訓練用的 5 個關鍵數字"""
    if len(peaks_idx) >= 3:
        periods = np.diff(time_stamps[peaks_idx])
        mean_period = np.mean(periods)
        period_cv = np.std(periods) / mean_period
    else:
        mean_period, period_cv = 0.0, 0.0

    if len(peaks_idx) >= 2:
        final_resting = np.mean(angles_smooth[-10:]) if len(angles_smooth) >= 10 else 0
        centered = angles_smooth - final_resting
        peak_amplitudes = np.abs(centered[peaks_idx])
        max_amplitude = np.max(peak_amplitudes)
        slope, _ = np.polyfit(np.arange(len(peak_amplitudes)), peak_amplitudes, 1)
    else:
        max_amplitude, slope = 0.0, 0.0

    mean_risk = np.mean(risk_index) if risk_index is not None else 0.0
    return [max_amplitude, mean_period, period_cv, slope, mean_risk]

def main():
    video_path = "test.mp4"  # 請替換為你的影片路徑
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"錯誤：無法開啟影片 '{video_path}'")
        return

    ret, frame = cap.read()
    if not ret: return

    print("請框選不倒翁的「頂部」特徵 (按 SPACE/ENTER 確認)")
    cv2.namedWindow("Select TOP", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select TOP", 800, 600)
    bbox_top = cv2.selectROI("Select TOP", frame, False)
    cv2.destroyWindow("Select TOP")

    print("請框選不倒翁的「底部」特徵 (按 SPACE/ENTER 確認)")
    cv2.namedWindow("Select BOTTOM", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select BOTTOM", 800, 600)
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

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    angles, time_stamps = [], []
    frame_id = 0

    cv2.namedWindow("Physics Angle Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Physics Angle Tracking", 800, 600)

    print("開始追蹤，按 ESC 結束...")
    while True:
        ret, frame = cap.read()
        if not ret: break

        success_top, bbox_top = tracker_top.update(frame)
        success_bottom, bbox_bottom = tracker_bottom.update(frame)

        if success_top and success_bottom:
            cx_top = bbox_top[0] + bbox_top[2]/2
            cy_top = bbox_top[1] + bbox_top[3]/2
            cx_bottom = bbox_bottom[0] + bbox_bottom[2]/2
            cy_bottom = bbox_bottom[1] + bbox_bottom[3]/2

            cv2.circle(frame, (int(cx_top), int(cy_top)), 5, (0, 0, 255), -1)
            cv2.circle(frame, (int(cx_bottom), int(cy_bottom)), 5, (255, 0, 0), -1)
            cv2.line(frame, (int(cx_bottom), int(cy_bottom)), (int(cx_top), int(cy_top)), (0, 255, 0), 2)

            dx = cx_top - cx_bottom
            dy = cy_bottom - cy_top
            angle = math.degrees(math.atan2(dx, dy))
            
            angles.append(angle)
            time_stamps.append(frame_id / fps)

            cv2.putText(frame, f"Angle: {angle:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Physics Angle Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    angles = np.array(angles)
    time_stamps = np.array(time_stamps)

    angles_smooth, angle_eq, peaks_idx, risk_index = analyze_roly_poly_physics(time_stamps, angles)

    # ==========================================
    # 【重點新增】印出機器學習專用的陣列格式
    # ==========================================
    features = []
    if angles_smooth is not None:
        features = extract_ml_features(time_stamps, angles_smooth, peaks_idx, risk_index)
        print("\n=======================================================")
        print("🤖 [AI 訓練特徵萃取完成]")
        print("格式: [Max_Amplitude, Mean_Period, Period_CV, Decay_Slope, Mean_Risk]")
        print(f"數據: [{features[0]:.4f}, {features[1]:.4f}, {features[2]:.4f}, {features[3]:.4f}, {features[4]:.4f}]")
        print("=======================================================\n")

    if angles_smooth is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        ax1.plot(time_stamps, angles, color='gray', alpha=0.3, label="Raw Angle")
        ax1.plot(time_stamps, angles_smooth, color='blue', linewidth=2, label="Filtered Angle")
        ax1.axhline(angle_eq, color='green', linestyle='--', label=f"Resting Eq ({angle_eq:.1f}°)")

        if len(peaks_idx) > 0:
            ax1.scatter(time_stamps[peaks_idx], angles_smooth[peaks_idx], color='red', zorder=5, label="Detected Peaks")
        
        ax1.set_title("Roly-Poly Motion Analysis: Oscillation & Decay")
        ax1.set_ylabel("Tilt Angle (Degrees)")
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2.plot(time_stamps, risk_index, color='purple', linewidth=2, label="Tipping Risk Index (0-100)")
        ax2.fill_between(time_stamps, risk_index, 80, where=(risk_index >= 80), facecolor='red', alpha=0.3, interpolate=True)
        ax2.fill_between(time_stamps, risk_index, 50, where=((risk_index >= 50) & (risk_index < 80)), facecolor='orange', alpha=0.3, interpolate=True)
        
        ax2.axhline(80, color='red', linestyle='-.', alpha=0.6, label="Critical Threshold (80)")
        ax2.axhline(50, color='orange', linestyle=':', alpha=0.6, label="Warning Threshold (50)")

        ax2.set_title("Dynamic Stability Assessment & Tipping Risk Prediction")
        ax2.set_xlabel("Time (Seconds)")
        ax2.set_ylabel("Risk Index (%)")
        ax2.set_ylim(0, 105)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()

    # ==========================================
    # 【重點新增】互動式存入 CSV (自動化建立資料集)
    # ==========================================
    if angles_smooth is not None and len(features) == 5:
        print("💾 [建立機器學習資料庫]")
        user_input = input("👉 若圖表無誤，請輸入【真實重量(克)】並按 Enter 存檔 (若追蹤失敗請直接按 Enter 略過): ")
        
        if user_input.strip().replace('.', '', 1).isdigit(): 
            weight = float(user_input.strip())
            csv_filename = "rolypoly_dataset.csv"
            file_exists = os.path.isfile(csv_filename)
            
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Video_Name", "Max_Amplitude", "Mean_Period", "Period_CV", "Decay_Slope", "Mean_Risk", "True_Weight_g"])
                
                row_data = [video_path] + [round(f, 4) for f in features] + [weight]
                writer.writerow(row_data)
            print(f"✅ 成功將數據寫入 {csv_filename}！")
        else:
            print("🚫 略過存檔。")

if __name__ == "__main__":
    main()