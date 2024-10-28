# from squash import Functions
# normal_ball_pos=[ [458, 192, 243], [461, 203, 244], [465, 214, 245], [469, 226, 246], [473, 239, 247], [477, 255, 248]]
# print(str(Functions.is_ball_false_pos(normal_ball_pos))) # should be false
# false_pos=[[458,192,400], [458,192,244], [458,192,245], [458,192,246], [458,192,248], [458,192,250], [458,192,254], [458,192,255]]
# print(str(Functions.is_ball_false_pos(false_pos))) # should be true
from squash import Referencepoints
refrence_points=Referencepoints.get_refrence_points('random.mp4', 640, 360)
print(refrence_points)