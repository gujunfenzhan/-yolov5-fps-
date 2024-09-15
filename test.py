import time
import win32api
import win32con

fov_horizontal = 1380
fov_vertical = fov_horizontal * 9 / 13.8
time.sleep(2)
target_x = 1920
target_y = 0
cross_hair_x = 960
cross_hair_y = 540
x_move = (target_x-cross_hair_x) / 1920 * fov_horizontal
y_move = (target_y-cross_hair_y) / 1080 * fov_vertical
print(x_move)
print(y_move)
time.sleep(2)
win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x_move), int(y_move))