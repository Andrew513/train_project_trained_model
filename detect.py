#inputting class_id x1 y1 x2 y2 
#x1 x2 y1 y2 are absolute coordinates

#the original yolo label are relative coordinate
#img_width*center_x[i]-img_width*x_width/2

#assuming the img has width 640 and height 480
from collections import deque
import sqlite3

conn = sqlite3.connect('boundingBox.db')
c= conn.cursor()

c.execute("SELECT * FROM bbox")
items = c.fetchall()

obstacle_detection = deque([0,0,0])
def obstacle_check():
    obstacle_detection.appendleft(1)
    obstacle_detection.pop()
    x_sum = 0;
    for x in obstacle_detection:
        x_sum += x 
    if x-sum == 3:
        print("obstacle detected")

# img_width=640
# img_height=480
classId = []
top_left_x = []
top_left_y = []
bot_right_x = []
bot_right_y = []

#open file 

for item in items:
    classId.append(item[0])
    top_left_x.append(item[1])
    top_left_y.append(item[2])
    bot_right_x.append(item[3])
    bot_right_y.append(item[4])
#knowing which is rail and get their bounding box's axis
#just for backup plan
#if there are multiple rails(detected other rail)
#assuming our camera is in the center of the road
# if classId ==80 and 0.4<= top_left_x <= 0.6?
for i in range (len(classId)):
    if classId[i] == 80:
        rail_right_x = bot_right_x[i]
        rail_left_x = top_left_x[i]
        rail_top_y = top_left_y[i]
        rail_bot_y = bot_right_y[i]

# #if non-rail stuff on the way
for i in range (len(classId)):
    if classId[i] != 80:
        item_right_x = bot_right_x[i]
        item_left_x = top_left_x[i]
        item_top_y = top_left_y[i]
        item_bot_y = bot_right_y[i]
        #scenario 1
        if item_left_x<=rail_left_x and item_right_x<=rail_right_x and item_right_x>=rail_left_x and item_bot_y >= rail_bot_y and item_bot_y <= rail_bot_y :
            obstacle_check()
        #scenario 2  
        if item_left_x>=rail_left_x and item_right_x>=rail_right_x and item_left_x<=rail_right_x and item_bot_y >= rail_bot_y and item_bot_y <= rail_bot_y :
            obstacle_check()
        #scenario 3
        if item_left_x>=rail_left_x and item_right_x<=rail_right_x and item_bot_y >= rail_bot_y and item_bot_y <= rail_bot_y :
            obstacle_check()
        #scenario 4
        if item_left_x<=rail_left_x and item_right_x>=rail_right_x and item_bot_y >= rail_bot_y and item_bot_y <= rail_bot_y :
            obstacle_check()

#save consecutive changing curve