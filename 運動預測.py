import cv2
import numpy as np
from scipy.optimize import curve_fit

# ==========================================
# 模組三：運動預測模型 (二次多項式)
# ==========================================
def motion_model(t, a, b, c):
    """假設短時間內受力恆定，運動軌跡近似為二次多項式"""
    return a * t**2 + b * t + c

def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
    """繪製虛線的輔助函式"""
    dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = [int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), 
                 int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes)]
        end = [int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), 
               int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes)]
        cv2.line(img, tuple(start), tuple(end), color, thickness)

def main():
    video_path = 'test_video.mp4' # 請替換為您的影片路徑
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("無法開啟影片！")
        return

    ret, frame = cap.read()
    if not ret:
        return

    # ==========================================
    # 模組二：數據前處理 (比例尺轉換)
    # ==========================================
    print("請在畫面上框選「已知長度的參考物」（如直尺），按 Space 或 Enter 確認。")
    ref_bbox = cv2.selectROI("Select Reference Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Reference Object")
    
    # 假設框選的寬度代表真實世界的 0.1 公尺 (10公分)
    L_meter = 0.1  
    pixel_length = ref_bbox[2] # 取得框選寬度 (pixels)
    if pixel_length == 0:
        print("未選取參考物，程式結束。")
        return
        
    R = L_meter / pixel_length # 計算比例尺 R = L / pixel
    print(f"比例尺 R 計算完成: 1 pixel = {R:.6f} meters")

    # ==========================================
    # 模組一：影像擷取與追蹤 (CSRT)
    # ==========================================
    print("請在畫面上框選「欲追蹤的物體」，按 Space 或 Enter 確認。")
    track_bbox = cv2.selectROI("Select Target Object", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target Object")

    # 初始化 CSRT 追蹤器
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, track_bbox)

    # 用於儲存觀測數據
    t_data = []
    x_data = []
    y_data = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    # 預測參數設定
    N_frames_to_fit = 15  # 使用過去 N 幀進行擬合
    future_frames = 30    # 預測未來 30 幀的軌跡

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = frame_count / fps
        success, bbox = tracker.update(frame)
        
        if success:
            # 取得物體中心像素座標
            center_x = bbox[0] + bbox[2] / 2
            center_y = bbox[1] + bbox[3] / 2
            
            # 儲存數據 (將像素轉為公尺儲存，或保留像素用於繪圖)
            t_data.append(current_time)
            x_data.append(center_x)
            y_data.append(center_y)
            
            # 繪製物件 bounding box
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            
            # 繪製實際軌跡 (實線 - 紅色)
            for i in range(1, len(x_data)):
                cv2.line(frame, (int(x_data[i-1]), int(y_data[i-1])), 
                         (int(x_data[i]), int(y_data[i])), (0, 0, 255), 2)

            # ==========================================
            # 模組三：物理模型擬合與預測
            # ==========================================
            if len(t_data) >= 3: # 至少需要3個點才能擬合二次曲線
                # 取最近 N 幀的數據進行擬合
                fit_t = np.array(t_data[-N_frames_to_fit:])
                fit_x = np.array(x_data[-N_frames_to_fit:])
                fit_y = np.array(y_data[-N_frames_to_fit:])
                
                try:
                    # 最小平方法擬合 X 與 Y 方向的運動方程式
                    popt_x, _ = curve_fit(motion_model, fit_t, fit_x)
                    popt_y, _ = curve_fit(motion_model, fit_t, fit_y)
                    
                    # 軌跡外推
                    pred_pts = []
                    for i in range(1, future_frames + 1):
                        future_t = current_time + (i / fps)
                        pred_x = motion_model(future_t, *popt_x)
                        pred_y = motion_model(future_t, *popt_y)
                        pred_pts.append((int(pred_x), int(pred_y)))
                        
                    # 視覺化呈現 (虛線 - 綠色)
                    start_pt = (int(center_x), int(center_y))
                    for i in range(len(pred_pts)):
                        draw_dashed_line(frame, start_pt, pred_pts[i], (0, 255, 0), 2)
                        start_pt = pred_pts[i]
                        
                except Exception as e:
                    pass # 擬合失敗（例如數據完全共線）時跳過本次預測

        else:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow("Tracking and Prediction", frame)
        frame_count += 1
        
        # 按 'q' 鍵離開
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
