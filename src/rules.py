#rule_dict = {
#'SeatBase': ['B_WheelIntersection', 'F_WheelIntersection', 'CranksetCenter', 'HandleCenter'],
#'CranksetCenter': ['B_WheelIntersection', 'F_WheelIntersection', 'SeatBase'], 
#'HandleCenter': ['L_HandleTip', 'R_HandleTip', 'SeatBase'],
#'L_HandleTip': ['HandleCenter'],
#'R_HandleTip': ['HandleCenter'],
#'B_WheelIntersection': ['B_WheelCenter', 'B_WheelEnd', 'CranksetCenter'],
#'B_WheelEnd': ['B_WheelIntersection', 'B_WheelCenter'],
#'B_WheelCenter': ['B_WheelIntersection', 'B_WheelEnd'],
#'F_WheelIntersection': ['F_WheelCenter', 'F_WheelEnd', 'CranksetCenter', 'HandleCenter'], 
#'F_WheelEnd': ['F_WheelIntersection', 'F_WheelCenter'],
#'F_WheelCenter': ['F_WheelIntersection', 'F_WheelEnd'],
#}

rule_dict = {
'BackRest_Top_Left': ['BackRest_Top_Right', 'Seat_Left_Back'], 
'BackRest_Top_Right': ['BackRest_Top_Left', 'Seat_Right_Back'], 
'Leg_Left_Back': ['Seat_Left_Back'],
'Leg_Left_Front': ['Seat_Left_Front'], 
'Leg_Right_Back': ['Seat_Right_Back'], 
'Leg_Right_Front': ['Seat_Right_Front'],
'Seat_Left_Back': ['Leg_Left_Back', 'BackRest_Top_Left', 'Seat_Left_Front', 'Seat_Right_Back'], 
'Seat_Left_Front': ['Leg_Left_Front', 'Seat_Left_Back', 'Seat_Right_Front', ], 
'Seat_Right_Back': ['Leg_Right_Back', 'BackRest_Top_Right', 'Seat_Right_Front', 'Seat_Left_Back'],
'Seat_Right_Front': ['Leg_Right_Front', 'Seat_Right_Back', 'Seat_Left_Front']
}