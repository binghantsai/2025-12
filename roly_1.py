import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def analyze_v1_logic(time_stamps, x_coords):
    """
    第一版判定邏輯 (已淘汰)：
    單純依賴 X 軸位移，並使用「整段平均值」當作基準線。
    """
    if len(x_coords) < 10:
        print("數據過少，無法分析。")
        return None, None

    # 1. 基準線定位：直接算整段 X 座標的平均值 (這就是導致躺平誤判的元凶)
    baseline_x = np.mean(x_coords)
    
    # 2. 數據中心化
    centered_x = x_coords - baseline_x
    
    # 3. 尋找波峰 (向右極限與向左極限)
    pos_peaks, _ = signal.find_peaks(centered_x, height=5, distance=5)
    neg_peaks, _ = signal.find_peaks(-centered_x, height=5, distance=5)
    all_peaks_idx = np.sort(np.concatenate((pos_peaks, neg_peaks)))
    
    # 4. 判斷條件 A：穿過基準線的次數 (零交越點)
    zero_crossings = np.where(np.diff(np.sign(centered_x)))[0]
    has_swing = len(zero_crossings) >= 3
    
    # 5. 判斷條件 B：振幅是否明顯衰減
    is_decaying = False
    slope = 0
    if len(all_peaks_idx) >= 2:
        peak_amplitudes = np.abs(centered_x[all_peaks_idx])
        slope, _ = np.polyfit(np.arange(len(peak_amplitudes)), peak_amplitudes, 1)
        # 第一版死板的阻尼設定：要求斜率必須小於 -0.1 (容易冤枉高品質不倒翁)
        is_decaying = slope < -0.1 

    # 綜合輸出報告
    print("\n===== 物理特徵分析報告 (V1 雛形版) =====")
    print(f"X 軸平均基準線 = {baseline_x:.2f} px")
    print(f"穿過基準線次數 = {len(zero_crossings)} 次")
    print(f"振幅衰減斜率 = {slope:.4f} (需 < -0.1)")

    if has_swing and is_decaying:
        print("\n👉 最終判定：【真不倒翁】")
    else:
        print("\n👉 最終判定：【非不倒翁】")
        if not has_swing:
            print("   - 未達足夠的搖晃次數")
        if not is_decaying:
            print("   - 未偵測到明顯的能量衰減 (或受平移干擾)")

    return centered_x, all_peaks_idx

def main():
    video_path = "test.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("錯誤：無法開啟影片。")
        return

    ret, frame = cap.read()
    if not ret: return

    # 第一版使用傳統的 Bounding Box (方形選框)
    print("請框選不倒翁 (按 SPACE/ENTER 確認)")
    bbox = cv2.selectROI("V1 Tracking", frame, False)
    cv2.destroyWindow("V1 Tracking")

    if bbox == (0,0,0,0): return

    # 初始化 CSRT 追蹤器 (這就是遇到傾斜會變形飄移的舊方法)
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    x_coords = []
    time_stamps = []
    frame_id = 0

    print("開始 V1 追蹤... (按 ESC 結束)")
    while True:
        ret, frame = cap.read()
        if not ret: break

        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            # 取框框的 X 軸中心點
            cx = x + w / 2.0
            
            x_coords.append(cx)
            time_stamps.append(frame_id / fps)

            # 畫出追蹤框與中心點
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, (int(cx), int(y + h/2)), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"X: {cx:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            print("警告：追蹤目標丟失！")
            break

        cv2.imshow("V1 Tracker", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    if len(x_coords) == 0: return
    x_coords = np.array(x_coords)
    time_stamps = np.array(time_stamps)

    # 執行第一版分析並繪圖
    centered_x, peaks_idx = analyze_v1_logic(time_stamps, x_coords)

    if centered_x is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(time_stamps, centered_x, color='blue', label="Centered X Position")
        plt.axhline(0, color='green', linestyle='--', label="Average Baseline")
        
        if len(peaks_idx) > 0:
            plt.scatter(time_stamps[peaks_idx], centered_x[peaks_idx], color='red', label="Peaks")
        
        plt.title("V1 Prototype: X-Axis Displacement Analysis")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("X Displacement (Pixels)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()