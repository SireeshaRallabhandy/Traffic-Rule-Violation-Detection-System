from openalpr import Alpr
from argparse import ArgumentParser
from openalpr import Alpr
from argparse import ArgumentParser
import cv2
import sys
import os
import csv
import datetime
from filter import filter_mask
import os
os.add_dll_directory("C:/openalpr_64/openalpr_64")

alpr = None

try:
    dst = os.getcwd() + '\cropedImages'  # destination to save the images
    if not os.path.exists(dst):
        os.mkdir(dst)


    fieldnames = ['ID', 'Plate number', 'Image Path', 'Date', 'Time','Confidence']
    csvFile = open('Car_data_3.csv', 'a') # file to save the data
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(fieldnames)
    alpr = Alpr("mx", "C:/openalpr_64/openalpr_64/openalpr.conf", "C:/openalpr_64/openalpr_64/runtime_data")
    alpr.set_top_n(7)
    alpr.set_default_region("wa")
    alpr.set_detect_region(False)

    if not alpr.is_loaded():
        print("Error loading OpenALPR")
    else:
        print("Using OpenALPR " + alpr.get_version())

        cap = cv2.VideoCapture("C:/Users/SireeshaRallabhandy/Downloads/1.webm")
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)
        print ('Training BG Subtractor...')
        cv2.namedWindow('op', cv2.WINDOW_NORMAL)
        cnt=0
        while True:
            ok,frame=cap.read()
            if not ok:
                sys.exit()
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                Rfilter = cv2.bilateralFilter(gray, 9, 75, 75)
                # Threshold image
                ret, filtered = cv2.threshold(Rfilter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                filtered = cv2.medianBlur(frame, 5)
                fg_mask = bg_subtractor.apply(filtered, None, 0.01)
                fg_mask = filter_mask(fg_mask)
                contours,hierarchy= cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cntr = []
                for contour in contours:
                    contourSize = cv2.contourArea(contour)
                    if (contourSize > int(100000)):
                        cntr.append(contour)
                        cnt+=1
                for ii in range(0, len(cntr)):
                    (x, y, w, h) = cv2.boundingRect(cntr[ii])
                    cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
                    cropedframe=frame[y:y+h, x:x+w]
                    filename=dst+'\\'+str(cnt)+'.jpg'
                    cv2.imwrite(filename,cropedframe) # write the image
                    ret, enc = cv2.imencode("*.bmp", cropedframe)
                    results = alpr.recognize_array(bytes(bytearray(enc)))
                    if results['results']:
                        print('License plate Detected and Recorded')
                        time = datetime.datetime.now().time()
                        date = datetime.datetime.now().date()
                        plateno = results['results'][0]['plate']
                        confidance=results['results'][0]['confidence']
                        print('Plate: ',plateno, 'Confidance: ',confidance)

                        csvWriter.writerow([cnt, plateno.encode('utf-8'), filename.encode('utf-8'), date, time,confidance]) # write the csv file
                cv2.imshow('op', frame)
                if cv2.waitKey(33) == 27:
                    break
                else:
                    pass
finally:
    if alpr:
        alpr.unload()
        """
"""
        alpr.set_top_n(7)
        alpr.set_default_region("wa")
        alpr.set_detect_region(False)
        jpeg_bytes = open(options.plate_image, "rb").read()
        results = alpr.recognize_array(jpeg_bytes)

        # Uncomment to see the full results structure
        # import pprint
        # pprint.pprint(results)

        print("Image size: %dx%d" %(results['img_width'], results['img_height']))
        print("Processing Time: %f" % results['processing_time_ms'])

        i = 0
        for plate in results['results']:
            i += 1
            print("Plate #%d" % i)
            print("   %12s %12s" % ("Plate", "Confidence"))
            for candidate in plate['candidates']:
                prefix = "-"
                if candidate['matches_template']:
                    prefix = "*"

                print("  %s %12s%12f" % (prefix, candidate['plate'], candidate['confidence']))
