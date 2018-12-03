#!/usr/bin/python3

import os
import imutils
import sys
import local_tools as lt
import numpy as np
import math
import cv2
import struct
import aurora
from collections import namedtuple, deque
from time import sleep
from datetime import datetime as dt
import analysisHelper as ah
from analysisHelper import (ctf, ctfs, roi, box, circle) 


ENSO_VER = {'major': 1, 'minor': 1 }
DT_LOG_FMT = '%m/%d/%Y %H:%M:%S'
DT_FILE_FMT = '%Y%m%d-%H%M%S'
PNG_COMP = 9
PNG_PARMS = (cv2.IMWRITE_PNG_COMPRESSION, PNG_COMP)
ARROW = {'up' : 2490368, 'left' : 2424832, 'right' : 2555904, 'down' : 2621440}

# CTF calculation parameters
DATA_TRIM       = 0.050       # % of data to remove from the histogram
CTF_ADJ_LIMIT   = 0.40
CTF_SPEC_LIMIT  = 0.35
CTF_WARNING_CUTOFF = .3



def init(uut, log_folder, build_sz, config):
    global UUT
    global LOG_FOLDER
    global BUILD_SZ
    global DEBUG_FOLDER

    global G_BLUR
    global CNTR_BOX_MIN
    global CNTR_BOX_MAX
    global CNTR_BOX_RATIO
    global CNTR_BOX_X_POS
    global CNTR_BOX_Y_POS
    global CNTR_BOX_POS_TOL
    global REF_OFFSET_X
    global REF_SAMP_OFFSET_X
    global REF_SAMP_SZ
    global PAT_OFFSET_X
    global PAT_OFFSET_Y
    global PAT_SZ
    global THRESHOLD_FACTOR
    global BUILD_RES_X
    global BUILD_RES_Y
    global DISP_WIN
    global DISP_RES_FULL
    global DISP_RES_SM
    global WIDTH
    global HEIGHT

    UUT = uut
    LOG_FOLDER = log_folder
    BUILD_SZ = build_sz
    DEBUG_FOLDER = LOG_FOLDER + 'debug/'
    if not os.path.isdir(DEBUG_FOLDER): os.mkdir(DEBUG_FOLDER)

    cfg = config['CTF_SETTINGS'][str(BUILD_SZ)]
    G_BLUR = cfg['G_BLUR']
    CNTR_BOX_MIN = cfg['CNTR_BOX_MIN']
    CNTR_BOX_MAX = cfg['CNTR_BOX_MAX']
    CNTR_BOX_RATIO = cfg['CNTR_BOX_RATIO']
    CNTR_BOX_X_POS = cfg['CNTR_BOX_X_POS']
    CNTR_BOX_Y_POS = cfg['CNTR_BOX_Y_POS']
    CNTR_BOX_POS_TOL = cfg['CNTR_BOX_POS_TOL']
    REF_OFFSET_X = cfg['REF_OFFSET_X']
    REF_SAMP_OFFSET_X = cfg['REF_SAMP_OFFSET_X']
    REF_SAMP_SZ = cfg['REF_SAMP_SZ']
    PAT_OFFSET_X = cfg['PAT_OFFSET_X']
    PAT_OFFSET_Y = cfg['PAT_OFFSET_Y']
    PAT_SZ = cfg['PAT_SZ']
    THRESHOLD_FACTOR = cfg['THRESHOLD_FACTOR']
    BUILD_RES_X = cfg['BUILD_RES_X']
    BUILD_RES_Y = cfg['BUILD_RES_Y']
    DISP_WIN = cfg['DISP_WIN']
    DISP_RES_FULL = cfg['DISP_RES_FULL']
    DISP_RES_SM = cfg['DISP_RES_SM']
    WIDTH = cfg['WIDTH']
    HEIGHT = cfg['HEIGHT']

    return
#

def distance(p1, p2):
    return ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**(.5)

def findCoord(xOffset, yOffset, ppmmX, ppmmY, refX, refY):
    #convert to mm
    xOffsetMM = xOffset/ppmmX
    yOffsetMM = yOffset/ppmmY

    #apply offset
    return (refX + xOffsetMM, refY + yOffsetMM)



def draw_lines2(cam):
    cv2.namedWindow(DISP_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISP_WIN, DISP_RES_FULL[0], DISP_RES_FULL[1])
    cam.start_capture()
    positions = ['Upper Left', 'Lower Left', 'Upper Right', 'Lower Right', 'Done']

    lineArray = []


    font = cv2.FONT_HERSHEY_SIMPLEX
    font_sz = 1
    font_thickness = 3
    font_loc = (25,100)

    key = -1
    i = 0
    ppmmX = 0
    ppmmY = 0
    refPointList = []
    projectedPointList = []

    offsetList = []

    label = ['Reference X Axis', 'Reference Y Axis', 'Upper Left', 'Upper Right', 'Lower Right', 'Lower Left']

    while True:

        if key == 27:
            break
        rotated = imutils.rotate(cam.grab_frame(), 180)
        img_base = rotated
        #cv2.cvtColor(start_image, cv2.COLOR_GRAY2BGR)

        cut = img_base.mean() * .4

        # semi adaptive binary threshold trial
        # make rois for the center box location limits
        img_blurred = cv2.GaussianBlur(img_base, (G_BLUR, G_BLUR), 0)
        cv2.imshow('blurred', img_blurred)


# cv2.THRESH_BINARY
# cv2.THRESH_BINARY_INV
# cv2.THRESH_TRUNC
# cv2.THRESH_TOZERO
# cv2.THRESH_TOZERO_INV





        img_thresh = (cv2.threshold(img_blurred, 20, 0, cv2.THRESH_TOZERO_INV)[1])


        start_image = img_base.copy()
        start_image = cv2.cvtColor(start_image, cv2.COLOR_GRAY2BGR)
        cv2.imshow('thresh', img_thresh)


        cnts = cv2.findContours(img_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


        points = []
        xOffset = 'unknown'
        yOffset = 'unknown'

        try:
            cnts = cnts[1]
            for c in cnts:
                try:
                    a = cv2.contourArea(c)
                    mt = cv2.moments(c)
                    cp = (int(mt['m10'] / mt['m00']), int(mt['m01'] / mt['m00']))
                    if a > 10000 and a < 20000:
                        cv2.drawContours(start_image, [c], -1, (0,255,0), 3)
                        cv2.circle(start_image, cp, 2, (0,0,122), 2)
                        points.append(cp)


                except:
                    xxxxxxx = ' '




            xOffset = min(points[0][0], points[1][0]) - max(points[0][0], points[1][0])
            yOffset = min(points[0][1], points[1][1]) - max(points[0][1], points[1][1])



            cv2.putText(start_image, str(points[0][0]) + ',' + str(points[0][1]), points[0],font, font_sz, (122,122,0), font_thickness) 
            cv2.putText(start_image, str(points[1][0]) + ',' + str(points[1][1]), points[1],font, font_sz, (255,0,0), font_thickness) 


        except Exception as e:
            #print('?')
            #print(e)
            ghjsafds= 'asdasdasd'
       # cv2.cvtColor(start_image, cv2.COLOR_GRAY2BGR)



        cv2.putText(start_image, 'xOffset ' + str(xOffset), (25, 100) ,font, font_sz, (0,123,50), font_thickness)
        cv2.putText(start_image, 'yOffset ' + str(yOffset), (25, 1000) ,font, font_sz, (0,123,50), font_thickness)
        cv2.putText(start_image, label[i], (1000, 100), font, font_sz, (12, 53, 23), font_thickness)

        cv2.imshow(DISP_WIN, start_image)
        key = cv2.waitKeyEx(1)
        #cv2.imshow('detected circles',cimg)


        if key == ord('p'):
            

            #points[0] -> one points (x,y) points[1] one point 
            xOffset = max(points[0][0], points[1][0]) - min(points[0][0], points[1][0])
            yOffset = max(points[0][1], points[1][1]) - min(points[0][1], points[1][1])
            if i == 0:
                ppmmX = distance(points[0], points[1])/4
            elif i == 1:
                ppmmY = distance(points[0], points[1])/4
            elif i == 2:

                ul = findCoord(-1 * xOffset, yOffset, ppmmX, ppmmY, 5, HEIGHT - 5 )
                print(ul)
            elif i == 3:

                ur = findCoord(xOffset, yOffset, ppmmX, ppmmY, WIDTH - 5, HEIGHT - 5)
                print(ur)

            elif i == 4:

                lr = findCoord(xOffset, -1* yOffset, ppmmX, ppmmY, WIDTH - 5, 5 )
                print(lr)
            elif i == 5:
                ll = findCoord(-1 * xOffset, -1 * yOffset, ppmmX, ppmmY, 5, 5 )
                print(ll)

                break

            i+=1


    print( 'R  + L rs = ' + str(distance(ur, lr)) + ' ls = ' + str(distance(ul, ll)))
    avg = (distance(ur, lr) + distance(ul, ll))/2
    yratio = avg/(HEIGHT - .001*BUILD_SZ *20*2)
    print( 'U+ B ts = ' + str(distance(ul, ur)) + ' bs = ' + str(distance(ll, lr)))


    avg = (distance(ul, ur) + distance(ll, lr))/2
    xratio = avg/(WIDTH - .001*BUILD_SZ *20 *2)
    cam.stop_capture()
    cv2.destroyAllWindows()

    print(xratio, yratio)

    return xratio,yratio



def draw_roi_set(h_image, h_rois):
    for item in h_rois.blk_refs:
        cv2.rectangle(h_image, item.ul, item.lr, (255, 255, 0), 2)
    for item in h_rois.wht_refs:
        cv2.rectangle(h_image, item.ul, item.lr, (0, 255, 255), 2)
    for item in h_rois.patterns:
        cv2.rectangle(h_image, item.ul, item.lr, (255, 0, 255), 2)
    return
#
def find_box(h_img):
    """
    """
    result = {}
    # semi adaptive binary threshold trial
    cut = h_img.mean() * THRESHOLD_FACTOR
    # make rois for the center box location limits
    ctr_loc = circle((CNTR_BOX_X_POS, CNTR_BOX_Y_POS), CNTR_BOX_POS_TOL)

    img_blurred = cv2.GaussianBlur(h_img, (G_BLUR, G_BLUR), 0)
    img_thresh = cv2.threshold(img_blurred, cut, 255, cv2.THRESH_BINARY)[1]

    # create copy for contour display and debug
    img_debug = h_img.copy()
    img_debug = cv2.cvtColor(img_debug, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_debug, ctr_loc.cp, ctr_loc.rad, (0, 255, 255), 2)

    # find contours in the thresholded image
    cnts = cv2.findContours(img_thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    for c in cnts:     # loop over the contours
        x, y, w, h = cv2.boundingRect(c)
        #print(w, h, (w / h))
        if CNTR_BOX_MIN <= w <= CNTR_BOX_MAX and \
           CNTR_BOX_MIN <= h <= CNTR_BOX_MAX and \
           CNTR_BOX_RATIO[0] <= ( w / h ) <= CNTR_BOX_RATIO[1]:
            mt = cv2.moments(c)
            cp = (int(mt['m10'] / mt['m00']), int(mt['m01'] / mt['m00']))
            err_pos = int(math.sqrt(abs(cp[0] - ctr_loc.cp[0])**2 +
                                    abs(cp[1] - ctr_loc.cp[1])**2))
            #print('\n(%4d, %4d) err = %6.2f\n' % (cp[0], cp[1], err_pos))
            result['valid'] =  err_pos <= CNTR_BOX_POS_TOL
            if result['valid']:
                clr = (0, 255, 0)
            else:
                clr = (0, 0, 255)

            cv2.circle(img_debug, cp, 3, clr, 3)
            cv2.line(img_debug, cp, ctr_loc.cp, clr, 1)
            cv2.drawContours(img_debug, [c], -1, clr, 3)

            result['centroid'] = cp
            result['aspect'] = (w / h)
            result['size'] = (w, h)
            result['error_pos'] = err_pos
            return result, img_debug
        #
    # next countour
    result['valid'] = False
    return result, img_debug
#
def calc_ctf(pos, b_ref, w_ref, pat, name_base = None):
    """
    """
    # start with set of 3 ROIs: light and dark reference and pattern
    # blur the white refernece region and get the sample average
    # reference = w_ref - b_ref
    # take the pattern histogram and trim the top/bottom x% of values

    # pattern data analysis
    bin_rng = [0, 256]
    num_bins = 256
    pat_hist = np.histogram(pat.ravel(), num_bins, bin_rng)
    hist = np.array(pat_hist[0])

    # trim the top and bottom % from the hist data
    trim = DATA_TRIM * hist.sum()
    z = 0
    for i in range(255,50,-1):
        z = z + hist[i]
        if z > trim:
            break
    w_pat = i
    z = 0
    for i in range(0, 200):
        z = z + hist[i]
        if z > trim:
            break
    b_pat = i

    # ref analysis
    b_val = b_ref.mean()
    w_ref_blur = cv2.GaussianBlur(w_ref, (G_BLUR, G_BLUR), 0)
    w_val = w_ref_blur.mean()

    # ctf calc
    h_ctf = ctf(pos, w_val, b_val, w_pat, b_pat)

    # data logging
    if name_base != None:
        parms = (cv2.IMWRITE_PNG_COMPRESSION, 0)
        h_csv = open(name_base + '.csv', 'w')

        #write pattern histogram data
        ps = 'trim,  %6.2f\nb_pat, %3d\nw_pat, %3d\n\n' % (DATA_TRIM, b_pat, w_pat)
        lt.log_str(h_csv, ps, False)
        lt.log_str(h_csv, 'BIN, COUNT\n', False)
        for i in range(0, len(pat_hist[0])):
            str = '%3d,%6d\n' % (i, pat_hist[0][i])
            lt.log_str(h_csv, str, False)
        cv2.imwrite(name_base + '_pat.png', pat, PNG_PARMS)
        cv2.imwrite(name_base + '_b_ref.png', b_ref, PNG_PARMS)
        cv2.imwrite(name_base + '_w_ref.png', w_ref, PNG_PARMS)
        cv2.imwrite(name_base + '_w_ref_blur.png', w_ref_blur, PNG_PARMS)
        h_csv.close()

    return h_ctf
#
def ctf_check_module(cam):
    """
    """
    NUM_POSITIONS = 13
    results = {}
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_sz = 2
    font_thickness = 5
    font_loc = (25,100)
    cv2.namedWindow(DISP_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISP_WIN, DISP_RES_FULL[0], DISP_RES_FULL[1])

    dt_in = dt.now()
    results['dt_start'] = dt.strftime(dt_in, DT_LOG_FMT)
    results['success'] = False
    results['msg'] = ''
    results['data'] = []
    key = -1
    cam.start_capture()
    id = 0
    disp_num_map()
    print('move camera to position %2d' % id)
    print('press "esc" to abort')
    print('press "p" to capture image')
    print('press "f" to skip CTF capture and force a position skip (only use for unreadable regions)')
    average = 0
    totalCtf = [deque(),deque(),deque(),deque()]
    while id < NUM_POSITIONS:
        node = {}
        font_clr = (255,255,0)
        txt = 'position %d' % id
        node['position'] = id
        count = [0,0,0,0]
        average = 0
        totalCtf = [deque(),deque(),deque(),deque()]
        key = -1
        while True:

            img_base = cam.grab_frame()

            #rotated image 180
            rotated = imutils.rotate(img_base, 180)
            reply, img_debug = find_box(rotated)



            if key == 27:   #esc
                return
            elif reply['valid']:
                node['centroid'] = reply
                rois = ah.make_roi_set(rotated, reply['centroid'], REF_OFFSET_X, REF_SAMP_OFFSET_X, REF_SAMP_SZ, PAT_OFFSET_Y, PAT_OFFSET_X, PAT_SZ)
                p = False
                if key == ord('p'):
                    draw_roi_set(img_debug, rois)
                    p = True

                vals = ctfs()
                for i in range(0,4):
                    b_ref = rois.blk_refs[i].roi
                    w_ref = rois.wht_refs[i].roi
                    pat = rois.patterns[i].roi
                    hndl = '%s_pos_%d_%d' % (DEBUG_FOLDER+UUT, id, i)
                    result = calc_ctf(i, b_ref, w_ref, pat)
                    
                    if p == True:
                        r = calc_ctf(i, b_ref, w_ref, pat, hndl)
                        r.ctf = np.mean(totalCtf[i])
                        vals.add_item(r)

                    ul = rois.patterns[i].ul
                    lr = rois.patterns[i].lr




                    if result.valid:
                        count[i] +=1
                        if count[i] > 74:
                            totalCtf[i].popleft()

                        totalCtf[i].append(result.ctf)
                        average = np.mean(totalCtf[i])


                    if average > CTF_ADJ_LIMIT:
                        clr = (0,255,0)
                    elif average > CTF_WARNING_CUTOFF :
                        clr = (0, 255, 255)
                    else:
                        clr = (0,0,255)

                    cv2.rectangle(img_debug, ul, lr, clr, 2)

                    cv2.putText(img_debug, str(average)[:5], (ul[0] + 5, ul[1] - 20) ,font, font_sz, clr, font_thickness)
                
                if p == True:
                    #p specific 
                    node['ctf'] = vals.json_data
                    
                    f_name = '%s_%d_debug.jpg' % (DEBUG_FOLDER+UUT, id)
                    cv2.imwrite(f_name, img_debug)   # save the debug image
                    f_name = '%s_FOCUS_%d.png' % (LOG_FOLDER+UUT, id)
                    print('%2d writing: %s' % (id, f_name))
                    cv2.imwrite(f_name, rotated) # save the raw image
                    node['image_file'] = f_name
                    results['data'].append(node)
                    id += 1
                    break
            elif key == ord('f'):
                vals = ctfs()
                for x in range (0,4):
                    temp_ctf = ctf(x,-1, -2, -3, -4)
                    temp_ctf.ctf = 0.001
                    vals.add_item(temp_ctf)

                node['ctf'] = vals.json_data  
                f_name = '%s_%d_debug.jpg' % (DEBUG_FOLDER+UUT, id)
                cv2.imwrite(f_name, img_debug)   # save the debug image
                f_name = '%s_FOCUS_%d.png' % (LOG_FOLDER+UUT, id)
                print('%2d writing: %s' % (id, f_name))
                cv2.imwrite(f_name, rotated) # save the raw image
                node['image_file'] = f_name
                results['data'].append(node)
                id+=1
                break
            else:
                average = 0
                count = [0,0,0,0]
                totalCtf = [deque(),deque(),deque(),deque()]

            cv2.putText(img_debug, txt, font_loc, font, font_sz,
                        font_clr, font_thickness)
            cv2.imshow(DISP_WIN, img_debug)
            key = cv2.waitKey(1)
    #

    if len(results['data']) == NUM_POSITIONS:
        results['success'] = True
    else:
        results['msg'] = 'incomplete test results, missing focus images'
        print(results['msg'])

    dt_el = dt.now() - dt_in
    results['dt_end'] = dt.strftime(dt.now(), DT_LOG_FMT)
    results['elapsed_time'] = dt_el.total_seconds()
    
    cam.stop_capture()
    cv2.destroyAllWindows()
    return results
#
def align_check_module(cam):
    """
    """
    results = {}
    NUM_POSITIONS = 8

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_sz = 2
    font_thickness = 5
    font_loc = (25,100)
    cv2.namedWindow(DISP_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISP_WIN, DISP_RES_SM[0], DISP_RES_SM[1])

    dt_in = dt.now()
    results['dt_start'] = dt.strftime(dt_in, DT_LOG_FMT)
    results['success'] = False
    results['msg'] = ''
    results['positions'] = []
    
    id = 5
    disp_num_map()
    print('move camera to position %2d' % id)
    print('"esc" to abort')
    print('press "p" to capture image')
    cam.start_capture()
    while id < NUM_POSITIONS + 5:
        #os.system('CLS')
        node = {}
        #loc = 'position_' + str(id)
        #results[loc] = {}
        node['position'] = id
        txt = 'position %d' % id
        while True:
            font_clr = (255,255,0)

            #rotated image 180
            rotated = imutils.rotate(cam.grab_frame(), 180)
            image = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

            cv2.putText(image, txt, font_loc, font, font_sz,
                        font_clr, font_thickness)
            cv2.imshow(DISP_WIN, image)
            key = cv2.waitKey(1)
            if key == 27:
                return
            elif key == ord('p') or key == ord('P'):
                #results[loc] = 'PASS'
                #font_clr = (0,255,0)
                break
            elif key == ord('f') or key == ord('F'):
                #results[loc] = 'FAIL'
                #font_clr = (0,0,255)
                #break
                pass
        f_name = '%s_ALIGN_%d.jpg' % (LOG_FOLDER+UUT, id)
        print('%2d writing: %s' % (id, f_name))
        #rotated image 180
        rotated = imutils.rotate(cam.grab_frame(), 180)


        image = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
        # cv2.putText(image, results[loc], font_loc, font, font_sz,
                    # font_clr, font_thickness)
        cv2.imwrite(f_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        #results[loc]['file'] = f_name
        node['image_file'] = f_name
        results['positions'].append(node)
        id += 1

    if len(results['positions']) == NUM_POSITIONS:
        results['success'] = True
    else:
        results['msg'] = 'incomplete test results, missing focus images'
        print(results['msg'])

    dt_el = dt.now() - dt_in
    results['dt_end'] = dt.strftime(dt.now(), DT_LOG_FMT)
    results['elapsed_time'] = dt_el.total_seconds()

    cam.stop_capture()
    cv2.destroyAllWindows()
    return results
#

def disp_num_map():
    print('        Position Map  ')
    print('    12       5       6')
    print('         4       1    ')
    print('    11       0       7')
    print('         3       2    ')
    print('    10       9       8')
#
def adjust_shutter(cam):
    step = 0.10
    cv2.namedWindow(DISP_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISP_WIN, DISP_RES_FULL[0], DISP_RES_FULL[1])
    cam.start_capture()

    while True:
        auto, shutter = cam.shutter
        print('%s, %10.5f\r' % (str(auto), shutter), end='')
        #rotated image 180
        rotated = imutils.rotate(cam.grab_frame(), 180)


        cv2.imshow(DISP_WIN, rotated)
        key = cv2.waitKeyEx(1)
        if key == 27: # escape
            break
        elif key == ARROW['right']:
            cam.shutter = (shutter + step)
        elif key == ARROW['left']:
            cam.shutter = (shutter - step)
        elif key == ARROW['up']:
            cam.shutter = (shutter + (step * 10))
        elif key == ARROW['down']:
            cam.shutter = (shutter - (step * 10))

    cam.stop_capture()
    cv2.destroyAllWindows()
#
def adjust_focus_proc(proj, cam, final = False):
    MAX_STEP_SZ = 256
    cv2.namedWindow(DISP_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISP_WIN, DISP_RES_FULL[0], DISP_RES_FULL[1])
    cam.start_capture()

    step_sz = 1
    check_focus.count = [0,0,0,0]
    check_focus.totalCtf = [deque(),deque(),deque(),deque()]
    while True:
        focus_pos = proj.focus_position
        print('STEP SIZE: %4d, FOCUS POSITION: = %6d\r' % (step_sz, focus_pos), end='')
        if final:
            cv2.imshow(DISP_WIN, check_focus(cam, True))
        else:
            #rotated image 180
            rotated = imutils.rotate(cam.grab_frame(), 180)
            cv2.imshow(DISP_WIN, rotated)

        key = cv2.waitKeyEx(1)
        if key == 27: # escape
            break
        elif key == ARROW['right']:
            proj.move_focus_motor(+step_sz)
            check_focus.count = [0,0,0,0]
            check_focus.totalCtf = [deque(),deque(),deque(),deque()]
        elif key == ARROW['left']:
            proj.move_focus_motor(-step_sz)
            check_focus.count = [0,0,0,0]
            check_focus.totalCtf = [deque(),deque(),deque(),deque()]
        elif key == ARROW['up']:
            if step_sz < MAX_STEP_SZ: step_sz <<= 1
        elif key == ARROW['down']:
            if step_sz > 1: step_sz >>= 1
        elif key == ord('s') or key == ord('S'):
            str_dt = dt.strftime(dt.now(), DT_FILE_FMT)
            f_name = '%sdebug_%s.jpg' % (DEBUG_FOLDER, str_dt)

            #rotated image 180
            rotated = imutils.rotate(cam.grab_frame(), 180)
            cv2.imwrite(f_name, rotated)

    cam.stop_capture()
    cv2.destroyAllWindows()

    return
#
def check_focus(cam, feedback = False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_sz = 2
    font_thickness = 5

    img_base = cam.grab_frame()
    #rotated image 180
    rotated = imutils.rotate(img_base, 180)
    
    reply, img_debug = find_box(rotated)
    if feedback:
        if reply['valid']:
            rois = ah.make_roi_set(rotated, reply['centroid'], REF_OFFSET_X, REF_SAMP_OFFSET_X, REF_SAMP_SZ, PAT_OFFSET_Y, PAT_OFFSET_X, PAT_SZ)
            #ctfs = {}
            #vals = ctfs()
            for i in range(0,4):
                b_ref = rois.blk_refs[i].roi
                w_ref = rois.wht_refs[i].roi
                pat = rois.patterns[i].roi
                ul = rois.patterns[i].ul
                lr = rois.patterns[i].lr
                result = calc_ctf(i, b_ref, w_ref, pat)
                


                

                if result.valid:
                    check_focus.count[i] +=1
                    if check_focus.count[i] > 74:
                        check_focus.totalCtf[i].popleft()

                    check_focus.totalCtf[i].append(result.ctf)
                    average = np.mean(check_focus.totalCtf[i])

                if average > CTF_ADJ_LIMIT:
                    clr = (0,255,0)
                elif average > CTF_WARNING_CUTOFF :
                    clr = (0, 255, 255)
                else:
                    clr = (0,0,255)

                cv2.rectangle(img_debug, ul, lr, clr, 2)

                cv2.putText(img_debug, str(average)[:5], (ul[0] + 5, ul[1] - 20) ,font, font_sz, clr, font_thickness)
        else:
            check_focus.count = [0,0,0,0]
            check_focus.totalCtf = [deque(),deque(),deque(),deque()]

    else:
        check_focus.count = [0,0,0,0]
        check_focus.totalCtf = [deque(),deque(),deque(),deque()]
    return img_debug
#





