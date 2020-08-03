import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt
import math
from os.path import isfile, join
import time

np.set_printoptions(suppress=True)

RED = [0,0,255]
count = 760

class Index:

    def __init__(self, m, n):
        self.m = m
        self.n = n

def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        #print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv.VideoWriter(pathOut, cv.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def makeVideo():
    # pathIn = './final_out/'
    # pathOut = 'video.mov'
    # fps = 0.5
    # convert_frames_to_video(pathIn, pathOut, fps)

    pathIn = './'
    pathOut = 'output.mov'
    fps = 50
    convert_frames_to_video(pathIn, pathOut, fps)

def extractFrames(pathIn, pathOut):
    os.mkdir(pathOut)

    cap = cv.VideoCapture(pathIn)
    count = 0

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            #print('Read %d frame: ' % count, ret)
            cv.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    print(count)

def getCorrelationAtIndex(m,n,M,N,t,r):

    sub = t[n:n+N,m:m+M]
    # print(r.shape)
    # print(sub.shape)
    # np.asmatrix(sub)
    # np.asmatrix(r)

    c = np.multiply(r,sub)
    sum = np.sum(c)

    # sqr_t = np.square(t)
    # sum_t = np.sum(sqr_t)
    # sqr_r = np.square(r)
    # sum_r = np.sum(sqr_r)
    # corr = float(sum / np.sqrt(sum_t * sum_r))

    # c = 0.0
    # sum_t = 0.0
    # sum_r = 0.0
    # #print(m,n,M,N)
    # # print(m,m+M)
    # # print(n,n+N)
    #
    # for j in range(M):
    #     for i in range(N):
    #         #print("index:",i,j)
    #         c += int(t[i+n][j+m])*int(r[i][j])
    #         sum_t += np.abs(int(t[i+n][j+m]))**2
    #
    #         #print(int(t[i][j])*int(r[i-m][j-n]))
    #         #print(sum)
    #
    # for j in range(M):
    #     for i in range(N):
    #         sum_r += np.abs(int(r[i][j]))**2
    #
    # sum = float(c/np.sqrt(sum_t*sum_r))
    # #print(sum)


    return sum


def exhausiveSearch(test_img, ref_img, org_img):

    I = len(test_img[0])
    J = len(test_img)
    M = len(ref_img[0])
    N = len(ref_img)

    max_corr = -math.inf
    max_m = -1
    max_n = -1

    #corr = getCorrelationAtIndex(100, 100, M, N, t, r)
    # print(I-M)
    # print(J-N)

    srch_count = 0

    for m in range (int((I-M+1))):
        for n in range (int((J-N+1))):
            # print(J-N+1)
            # print(int(J-N+1)/2.0)

            #print("index:", m, n)
            corr = getCorrelationAtIndex(m,n,M,N,test_img,ref_img)
            #print(corr)
            if(corr > max_corr):
                max_corr = corr
                max_m = m
                max_n = n

            srch_count += 1

    #print(max_m,max_n)
    #rect = cv.rectangle(org_img, (max_m, max_n), (max_m + M, max_n + N), color=RED, thickness=2)
    # plt.imshow(rect, cmap="gray")
    # plt.show()

    return max_m, max_n, srch_count

def exhausiveSearchPwindow(m, n, test_img, ref_img, p):

    m1 = m - p
    m2 = m + p
    n1 = n - p
    n2 = n + p

    I = len(test_img[0])
    J = len(test_img)
    M = len(ref_img[0])
    N = len(ref_img)

    max_corr = -math.inf
    max_m = m
    max_n = n

    srch_count = 0

    for m in range(int(m1),int(m2 + 1)):
        for n in range(int(n1),int(n2 + 1)):
            # print(J-N+1)
            # print(int(J-N+1)/2.0)
            #print("index:", m, n)
            #print("hi")

            if(m + M >= I or n + N >= J or m < 0 or n < 0):
                #print("cont")
                continue

            corr = getCorrelationAtIndex(m, n, M, N, test_img, ref_img)
            #print("corr:",corr)
            #print("index inside if:", m, n)
            if (corr > max_corr):
                max_corr = corr
                max_m = m
                max_n = n

            srch_count += 1

            #print("max so far:",max_m,max_n)
    #rect = cv.rectangle(org_img, (max_m, max_n), (max_m + M, max_n + N), color=RED, thickness=2)
    # plt.imshow(rect, cmap="gray")
    # plt.show()

    return max_m, max_n, srch_count

def runExhaustive(out_file,p):

    ref_img = cv.imread('reference.jpg', 0)
    os.mkdir(out_file)
    os.chdir('data')

    #full exhaustive for first frame

    name_in = 'frame0.jpg'
    name_out = 'frame0.jpg'
    test_img = cv.imread(name_in, 0)
    org_img = cv.imread(name_in)

    I = len(test_img[0])
    J = len(test_img)
    M = len(ref_img[0])
    N = len(ref_img)

    m, n, srch_count = exhausiveSearch(test_img, ref_img, org_img)
    output_img = cv.rectangle(org_img, (m, n), (m + M, n + N), color=RED, thickness=2)
    #print("frame: 0 m:",m," n:",n)

    total_srch_count = srch_count

    os.chdir('..')
    cv.imwrite(os.path.join(out_file, name_out), output_img)

    for i in range(1, count):

        os.chdir('data')

        name_in = ('frame' + str(i) + '.jpg')
        name_out = ('frame' + str(i) + '.jpg')
        #print(name_in, name_out)

        test_img = cv.imread(name_in, 0)
        org_img = cv.imread(name_in)
        #print("yee",test_img)

        # print("J:", len(test_img))
        # print("I:", len(test_img[0]))
        # print("N:", len(ref_img))
        # print("M:", len(ref_img[0]))

        #window_img = test_img[n-p:n+p,m-p:m+p]
        #print("hhh",window_img.shape)
        m, n, srch_count = exhausiveSearchPwindow(m, n, test_img, ref_img, p)
        output_img = cv.rectangle(org_img, (m, n), (m + M, n + N), color=RED, thickness=2)
        #print("frame:",i," m:", m, " n:", n)

        total_srch_count += srch_count

        # cv.imwrite(out, output_img)
        os.chdir('..')
        cv.imwrite(os.path.join(out_file, name_out), output_img)

    return float(total_srch_count/count)

def logarithmicSearch(center_m, center_n, test_img, ref_img, p):

    srch_count = 0

    c = 1
    while 1:

        #print("Itn:",c)

        I = len(test_img[0])
        J = len(test_img)
        M = len(ref_img[0])
        N = len(ref_img)

        #print("init p:",p*2)
        k = np.ceil(np.log2(p*2))
        new_p = np.around(p)

        d = int(np.power(2,k-1))
        #print("gap:",new_p,"k:",k,"d:",d)

        ms = [center_m, center_m-d, center_m+d, center_m, center_m, center_m-d, center_m+d, center_m-d, center_m+d]
        ns = [center_n, center_n-d, center_n+d, center_n-d, center_n+d, center_n, center_n, center_n+d, center_n-d]

        max_corr = -math.inf
        max_m = center_m
        max_n = center_n

        for i in range(len(ms)):
            #print("hi")

            m = ms[i]
            n = ns[i]

            if (m + M >= I or n + N >= J or m < 0 or n < 0):
                #print("cont")
                continue

            corr = getCorrelationAtIndex(m, n, M, N, test_img, ref_img)
            #print(corr)
            if (corr > max_corr):
                max_corr = corr
                max_m = m
                max_n = n

            srch_count += 1

        center_m = max_m
        center_n = max_n
        p = np.around(p/2.0)
        c += 1

        if (d == 1):
            break


        # origin_m = center_m - d
        # origin_n = center_n - d
        #
        # max_corr = -math.inf
        # max_m = -1
        # max_n = -1
        #
        # for m in range(origin_m, origin_m + 2*d + 1, d):
        #     for n in range(origin_n, origin_n + 2*d + 1, d):
        #
        #         if (m + M >= I or n + N >= J or m < 0 or n < 0):
        #             continue
        #
        #         corr = getCorrelationAtIndex(m, n, M, N, test_img, ref_img)
        #         # print(corr)
        #         if (corr > max_corr):
        #             max_corr = corr
        #             max_m = m
        #             max_n = n
    #print("final pt",center_m,center_n)
    #rect = cv.rectangle(org_img, (center_m, center_n), (center_m + M, center_n + N), color=RED, thickness=2)
    return center_m, center_n, srch_count


def runLogarithmic(out_file, p):

    ref_img = cv.imread('reference.jpg', 0)
    os.mkdir(out_file)
    os.chdir('data')

    # full exhaustive for first frame

    name_in = 'frame0.jpg'
    name_out = 'frame0.jpg'
    test_img = cv.imread(name_in, 0)
    org_img = cv.imread(name_in)

    I = len(test_img[0])
    J = len(test_img)
    M = len(ref_img[0])
    N = len(ref_img)

    m, n, srch_count = exhausiveSearch(test_img, ref_img, org_img)
    output_img = cv.rectangle(org_img, (m, n), (m + M, n + N), color=RED, thickness=2)
    #print("frame:0 m:", m, " n:", n)

    total_srch_count = srch_count

    os.chdir('..')
    cv.imwrite(os.path.join(out_file, name_out), output_img)

    for i in range(1, count):
        os.chdir('data')

        name_in = ('frame' + str(i) + '.jpg')
        name_out = ('frame' + str(i) + '.jpg')
        # print(name_in, name_out)

        test_img = cv.imread(name_in, 0)
        org_img = cv.imread(name_in)

        m, n, srch_count = logarithmicSearch(m, n, test_img, ref_img, p)
        output_img = cv.rectangle(org_img, (m, n), (m + M, n + N), color=RED, thickness=2)
        #print("frame:", i, " m:", m, " n:", n)

        total_srch_count += srch_count

        # cv.imwrite(out, output_img)
        os.chdir('..')
        cv.imwrite(os.path.join(out_file, name_out), output_img)

    return float(total_srch_count/count)


def hierarchichalSearch(m, n, test_img, ref_img, org_img, p, level):

    I = len(test_img[0])
    J = len(test_img)
    M = len(ref_img[0])
    N = len(ref_img)

    test_img_dict = {}
    ref_img_dict = {}

    test_img_dict[0] = test_img
    ref_img_dict[0] = ref_img

    #print("org size:",test_img.shape, ref_img.shape)

    x = m
    y = n

    temp_test = test_img.copy()
    temp_ref = ref_img.copy()

    for l in range(1,level+1):

        lowpassed_test = cv.blur(temp_test, (3,3))
        lowpassed_ref = cv.blur(temp_ref, (3,3))

        resized_test = cv.resize(lowpassed_test, (int(lowpassed_test.shape[1]/2.0),int(lowpassed_test.shape[0]/2.0)))
        resized_ref = cv.resize(lowpassed_ref, (int(lowpassed_ref.shape[1]/2.0), int(lowpassed_ref.shape[0]/2.0)))

        # print(temp_test.shape)
        # print(temp_ref.shape)
        # print(resized_test.shape)
        # print(resized_ref.shape)

        test_img_dict[l] = resized_test
        ref_img_dict[l] = resized_ref

        temp_test = resized_test.copy()
        temp_ref = resized_ref.copy()

    srch_count = 0

    for l in range(level, -1, -1):

        #print(test_img_dict[l].shape)
        #print(ref_img_dict[l].shape)

        temp_m = int(x/np.power(2,l))
        temp_n = int(y/np.power(2,l))

        #print("level:",l)
        #print("cnter:",temp_m,temp_n)


        if(l == level):

            #print("area size", int(p / (np.power(2, l))))

            #temp_m = int(m/(np.power(2,l)))
            #temp_n = int(n/(np.power(2,l)))

            #print("center of smallest",temp_m,temp_n)

            m, n, s_count = exhausiveSearchPwindow(temp_m, temp_n, test_img_dict[l], ref_img_dict[l], np.ceil(p/(np.power(2,l)))) #only half gap goes, so send p/4 for level 2
            #m, n, s_count = logarithmicSearch(temp_m, temp_n, test_img_dict[l], ref_img_dict[l], np.ceil(p/np.power(2,l)))

            srch_count += s_count
            #print("level",l,"mn once",m,n)

        else:
            #print(m,n,test_img_dict[l].shape,ref_img_dict[l].shape)

            m, n, s_count = exhausiveSearchPwindow(m, n, test_img_dict[l], ref_img_dict[l], 1)  # only half gap goes, so send p/4 for level 2
            #m, n, s_count = logarithmicSearch(m, n, test_img_dict[l], ref_img_dict[l], 1)

            srch_count += s_count
            #print("level",l," mn", m, n)

        if(l > 0):

            f_m = x/np.power(2,l-1)
            s_m = 2.0*(m - temp_m)

            #print("cm zm", c_m,z_m)

            f_n = y/np.power(2,l-1)
            s_n = 2.0*(n - temp_n)

            #print("cn zn", c_n, z_n)

            m = int(f_m + s_m)
            n = int(f_n + s_n)

            # m = int(x/np.power(2,l-1) + 2.0*(m - temp_m))
            # n = int(y/np.power(2,l-1) + 2.0*(n - temp_n))

            #print("new mn",m,n)
            #print("level image size",test_img_dict[l-1].shape)


    #print("final pt", m, n)
    #rect = cv.rectangle(org_img, (m, n), (m + M, n + N), color=RED, thickness=2)
    #print("ref frame extends upto",m+M, n+N)
    return m, n, srch_count


def runHeirarchichal(out_file, p, level):

    ref_img = cv.imread('reference.jpg', 0)
    os.mkdir(out_file)
    os.chdir('data')

    # full exhaustive for first frame

    name_in = 'frame0.jpg'
    name_out = 'frame0.jpg'
    test_img = cv.imread(name_in, 0)
    org_img = cv.imread(name_in)

    I = len(test_img[0])
    J = len(test_img)
    M = len(ref_img[0])
    N = len(ref_img)

    m, n, srch_count  = exhausiveSearch(test_img, ref_img, org_img)
    output_img = cv.rectangle(org_img, (m, n), (m + M, n + N), color=RED, thickness=2)
    #print("frame: 0 m:", m, " n:", n)

    total_srch_count = srch_count

    os.chdir('..')
    cv.imwrite(os.path.join(out_file, name_out), output_img)

    for i in range(1, count):

        os.chdir('data')

        name_in = ('frame' + str(i) + '.jpg')
        name_out = ('frame' + str(i) + '.jpg')
        # print(name_in, name_out)

        test_img = cv.imread(name_in, 0)
        org_img = cv.imread(name_in)

        m, n, srch_count = hierarchichalSearch(m, n, test_img, ref_img, org_img, p, level)
        output_img = cv.rectangle(org_img, (m, n), (m + M, n + N), color=RED, thickness=2)
        #print("frame:", i, " m:", m, " n:", n)

        total_srch_count += srch_count

        # cv.imwrite(out, output_img)
        os.chdir('..')
        cv.imwrite(os.path.join(out_file, name_out), output_img)

    return float(total_srch_count/count)


def main():

    #extractFrames('movie.mov', 'data')

    lst = [1,2,4,8,16,32,64,128,256,512]

    file = open("comparison.txt","w")
    # file.write("yoo")
    #file.write("P           Exhaustive          Logarithmic         Hierarchical\n")

    start = time.time()

    # print("p    exhaustive      logarithmic     hierarchical")
    #
    # all_arr = []
    #
    # for p in lst:
    #
    #     arr = []
    #     arr.append(p)
    #
    #     out_file = 'exhaust_out' + str(p)
    #     avg_srch_count_1 = runExhaustive(out_file,p)
    #     arr.append(avg_srch_count_1)
    #     #print("avg srch count for exhaustive, p =",p,":",avg_srch_count)
    #     os.chdir(out_file)
    #
    #     makeVideo()
    #     os.chdir('..')
    #
    #     out_file = 'logarithmic_out' + str(p)
    #     avg_srch_count_2 = runLogarithmic(out_file, p)
    #     arr.append(avg_srch_count_2)
    #     #print("avg srch count for logarithmic, p =", p, ":", avg_srch_count)
    #     os.chdir(out_file)
    #
    #     makeVideo()
    #     os.chdir('..')
    #
    #     out_file = 'hierarchical_out' + str(p)
    #     avg_srch_count_3 = runHeirarchichal(out_file, p, 2)
    #     arr.append(avg_srch_count_3)
    #     #print("avg srch count for hierarchical, p =", p, ":", avg_srch_count)
    #     os.chdir(out_file)
    #
    #     makeVideo()
    #     os.chdir('..')
    #
    #     print("p =",p,avg_srch_count_1,avg_srch_count_2,avg_srch_count_3)
    #     file.write("p =" + str(p) + str(avg_srch_count_1) + str(avg_srch_count_2) + str(avg_srch_count_3))
    #
    #     all_arr.append(arr)
    #
    # all_arr = np.array(all_arr)
    # np.save("output", all_arr)

    all_arr = np.load("output.npy")
    print(all_arr)

    np.savetxt("comparison.txt",all_arr,delimiter='\t',fmt='%f',header="    P                E              L               H",comments='')

    #plotting

    plt.plot(all_arr[:,0], all_arr[:,1], '-o', color='RED')
    plt.xlabel('p value')
    plt.ylabel('exhaustive: avg number of searches per frame')
    plt.show()

    plt.plot(all_arr[:, 0], all_arr[:, 2], '-o',color='GREEN')
    plt.xlabel('p value')
    plt.ylabel('logarithmic: avg number of searches per frame')
    plt.show()

    plt.plot(all_arr[:, 0], all_arr[:, 3], '-o',color='BLUE')
    plt.xlabel('p value')
    plt.ylabel('hierarchical: avg number of searches per frame')
    plt.show()

    end = time.time()
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    file.close()


if __name__ == "__main__":
    main()