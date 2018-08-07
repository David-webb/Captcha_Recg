#!/usr/bin/python
# -*- coding: utf-8 -*-

# Created by David Teng on 18-7-4

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
import scipy.spatial.distance as scidst
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
# import pytesseract
import cv2
import numpy
import Queue
import os
import math
import copy
import random
import pytesseract

class rmBKnoise():
    """
        sogou验证码尺寸: 44 * 140
        1. 去除背景小字母
        2. 寻找直线的两端
        3.
    """

    def rmbkalpahb(self, imgname, b_threshold=100, savepath="bwtest.jpg"):
        """ opencv: 去除背景小字母,灰度化之后再二值化 """
        im = cv2.imread(imgname)  # 读取图片
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
        retval, im_at_fixed = cv2.threshold(im_gray, b_threshold, 255, cv2.THRESH_BINARY) # 二值化
        # cv2.imshow('bw',im_at_fixed)
        # blur = cv2.blur(im_at_fixed, (5, 5))  # 去除噪声
        # c, h = cv2.findContours(im_at_fixed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(im_at_fixed, c, -1, (0, 0, 255), 3)
        # cv2.imshow("轮廓", im_at_fixed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print im_at_fixed
        cv2.imwrite(savepath, im_at_fixed)

    def rmcurselines(self):

        # endblock = img[0:44, 139:140]
        # print startblock, endblock
        # print startblock
        pass

    def updatestack(self, pixels, tmpstack, imgobj):

        pass

    def dealwithnoisepoint(self, imgobj):
        """ 清除噪点 4域 8域 泛洪"""
        # cv2.imread(fname)
        # cv2.floodFill()
        h, w = imgobj.shape
        # for r in h:
        pass


        pass

    def getsuspectblocks(self, indexofpixel, mode, imgobj):
        """
            获得任意一列(行)的可疑block
        :param indexofpixel: 下标
        :param mode: "Row" 或 "Col", 确定下标是行或列的下标
        :return:
        """
        def getblackarea(sampleblock):
            flag = False
            tmpstart = 0
            tmpend = 0
            ansblcok = []
            for i, p in enumerate(sampleblock):
                if p < 200:
                    if flag == False:
                        flag = True
                        tmpstart = i
                else:
                    if flag:
                        tmpend = i
                        flag = False
                        ansblcok.append([tmpstart, tmpend])  # 终止点为白点
            if flag:
                ansblcok.append([tmpstart, tmpend])
            return ansblcok

        h, w = imgobj.shape
        if mode == "R":
            assert indexofpixel >= 0 and indexofpixel < h
            sus_blocklist = imgobj[indexofpixel, :]
            return getblackarea(sus_blocklist)
        elif mode == "C":
            assert indexofpixel >= 0 and indexofpixel < w
            sus_blocklist = imgobj[:, indexofpixel]
            return getblackarea(sus_blocklist)
        else:
            # raise # 抛出异常
            return []   #

    def detectnextpixel(self, startblock, imgobj, savepath=""):
        """ 获取选一个可疑像素的点 """
        if startblock == None:
            print "没有找到起始点...."
            return
        width = len(startblock)

        h, w = imgobj.shape

        # print h, w
        tmpstack = Queue.Queue()
        for i in startblock:
            tmpstack.put(i)

        def crossarea(width, point, imgobj, getsusblockfunc, threshold=35):
            """
            统计以width为半径,以point为中心的矩形区域中黑点个数,设置阈值,小于阈值就抹除该点
                :param width: 曲线的宽度
                :param point: 当前待处理的点(是否抹除)
                :param imgobj: 图像矩阵
                :param threshold: 密度阈值,用于判断可疑点附近的黑点密度是否达到抹除条件,阈值越大,可疑点被抹除的可能性越大
                :return: True:抹除,False:不抹除
            """
            h, w = imgobj.shape
            r, c = point
            # blocklists = getsusblockfunc(indexofpixel=c, mode="C", imgobj=imgobj)
            # b = [block for block in blocklists if r in range(block[0], block[1])]
            # # return False if (b[1] - b[0]) > 5 else True
            # if b:
            #     b = b[0]
            #     if (b[1] - b[0]) < width: return True
            # else:
            #     # print width
            rs = 0 if r - width < 0 else r - width
            re = h - 1 if r + width > h - 1 else r + width
            # rs = 0 if r - 2*width < 0 else r - width
            # re = h - 1 if r + 2*width > h - 1 else r + width
            cs = 0 if c - width < 0 else c - width
            ce = w - 1 if c + width > w - 1 else c + width
            safesquare = imgobj[rs:re, cs:ce]
            # threshold = int(4 * (width ** 2) * 0.5)
            threshold = int(4 * (width ** 2) * 0.4)
            return True if len(safesquare[safesquare < 200]) < threshold else False

        def putornot(r, c, imobj, tmpstack, scanset):
            # print r, c, imobj[r, c]
            if imobj[r, c] < 200:
                if crossarea(width, (r, c), imgobj, self.getsuspectblocks):
                    imobj[r, c] = 255
                if (r, c) not in scanset:
                    tmpstack.put((r, c))
                    scanset.add((r, c))
            pass

        scanset = set()
        while not tmpstack.empty():
            p = tmpstack.get()
            r, c = p
            # print 'h', r, c
            if r > 0 and r < h-1:      # 这个点在中间
                if c < w-1:
                    # putornot(r - 1, c, imgobj, tmpstack, scanset)       # 当前点的上一个点
                    # putornot(r+1, c, imgobj, tmpstack, scanset)           # 当前点的下一个点
                    putornot( r - 1, c + 1, imgobj, tmpstack, scanset)
                    putornot(r, c + 1, imgobj, tmpstack, scanset)
                    putornot(r + 1, c + 1, imgobj, tmpstack, scanset)
            elif r == 0:    # 第一行
                if c < w-1:     # 不是最后一列
                    putornot(r, c+1, imgobj, tmpstack, scanset)
                    putornot(r + 1, c+1, imgobj, tmpstack, scanset)
                    # putornot(r + 1, c, imgobj, tmpstack, scanset)  # 当前点的下一个点
            elif r == h-1:  # 最后一行
                if c < w-1:     # 不是最后一列
                    putornot(r - 1, c+1, imgobj, tmpstack, scanset)
                    putornot(r, c+1, imgobj, tmpstack, scanset)
                    # putornot(r - 1, c, imgobj, tmpstack, scanset)  # 当前点的上一个点
        cv2.imwrite(savepath, imgobj)
        pass

    def delextrackstartblock(self, startblock):
        for block in startblock:
            return [(i, 0) for i in range(block[0], block[1])]

    def getstartendblock(self, imgname='bwtest.jpg', imgobj=None):
        """ 获取图像起始列和终止列的可疑 black_block """
        if imgobj:
            img = imgobj
        else:
            img = cv2.imread(imgname, 0)
        # print img.shape
        startblock = img[0:44, 0]
        flag = False
        tmpstart = 0
        tmpend = 0
        ansblcok = []
        for i, p in enumerate(startblock):
            if p < 200:
                if flag == False:
                    flag = True
                    tmpstart = i
            else:
                if flag:
                    tmpend = i
                    flag = False
                    ansblcok.append([tmpstart, tmpend])  # 终止点为白点
        if flag:
            ansblcok.append([tmpstart, tmpend])
        ansblcok = [b for b in ansblcok if (b[1]-b[0])>2]
        return self.delextrackstartblock(ansblcok), img
        pass

    def run(self):
        """ """
        files = os.listdir("sogoucapture")
        abpath = os.path.abspath("sogoucapture")
        for f in files:
            fname = os.path.join(abpath, f)
            self.rmbkalpahb(fname)
            startblock, imgobj = tobj.getstartendblock("bwtest.jpg")
            tobj.detectnextpixel(startblock, imgobj, "%s_c.jpg" % fname)
            print "end processing file %s......" % f
        pass


class rmcurselinewithscanning():
    """
        使用逐行(列)扫描的方式清除图片中的曲线
    """

    def __init__(self, imgpath):

        if not os.path.exists(imgpath):
            print "图片路径不存在, 退出执行..."
        self.imgpath = os.path.abspath(imgpath)
        self.fname = os.path.split(self.imgpath)[1]
        self.imgobj = None
        self.s_blocks = []
        pass


    def rmeightNeibornoisepoint(self):
        """ 用8领域法去除噪声点,特别是连续曲线中的噪声点 """
        def pixel_8_neibor(r, c, imgobj):
            black_count = 0
            if r-1>=0:
                black_count += 1 if imgobj[r-1,c] < 200 else 0  # 左上方
                if c-1>=0:
                    black_count += 1 if imgobj[r-1,c-1] < 200 else 0 # 正上方
                if c+1<w:
                    black_count += 1 if imgobj[r-1, c+1] < 200 else 0 # 右上方
            if r+1 < h:
                black_count += 1 if imgobj[r+1, c] < 200 else 0    # 正下方
                if c-1>=0:
                    black_count += 1 if imgobj[r+1, c-1] < 200 else 0 # 左下方
                if c+1<w:
                    black_count += 1 if imgobj[r+1, c+1] < 200 else 0 # 右下方

            if c-1 >= 0:
                black_count += 1 if imgobj[r, c-1] < 200 else 0  # 左侧 
            if c+1 < w:
                black_count += 1 if imgobj[r, c+1] < 200 else 0  # 右侧

            if black_count >= 5:
                imgobj[r,c] = 0
            else:
                imgobj[r,c] = 1

            pass
        h, w = self.imgobj.shape
        for r in range(h):
            for c in range(w):
                if self.imgobj[r,c] >= 200:
                    pixel_8_neibor(r,c,self.imgobj)
        pass

    def tobinarypic(self, b_threshold=100):
        """ 将验证码图片二值化 """
        im = cv2.imread(self.imgpath)  # 读取图片
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
        retval, im_at_fixed = cv2.threshold(im_gray, b_threshold, 255, cv2.THRESH_BINARY)  # 二值化

        self.imgobj = im_at_fixed
        return im_at_fixed

    def getsuspectblocks(self, indexofpixel, mode):
        """
            获得任意一列(行)的可疑block
        :param indexofpixel: 下标
        :param mode: "Row" 或 "Col", 确定下标是行或列的下标
        :return:
        """
        def getblackarea(sampleblock):
            flag = False
            tmpstart = 0
            tmpend = 0
            ansblcok = []
            for i, p in enumerate(sampleblock):
                if p < 200:
                    if flag == False:
                        flag = True
                        tmpstart = i
                else:
                    if flag:
                        tmpend = i
                        flag = False
                        ansblcok.append([tmpstart, tmpend])  # 终止点为白点
            if flag:
                ansblcok.append([tmpstart, tmpend])
            return ansblcok

        h, w = self.imgobj.shape
        if mode == "R":
            assert indexofpixel >= 0 and indexofpixel < h
            sus_blocklist = self.imgobj[indexofpixel, :]
            return getblackarea(sus_blocklist)
        elif mode == "C":
            assert indexofpixel >= 0 and indexofpixel < w
            sus_blocklist = self.imgobj[:, indexofpixel]
            return getblackarea(sus_blocklist)
        else:
            # raise # 抛出异常
            return []   #


    def calcblockdis(self, old_block, new_block):
        """ 计算前一行(列)和当前行(列)之间任意两个可疑block的距离 """
        oldlen = (old_block[1] - old_block[0])
        middle_point = old_block[0] + oldlen / 2
        newlen = new_block[1] - new_block[0]
        new_middle_point = new_block[0] + newlen / 2
        return abs(new_middle_point - middle_point)
        pass

    def getstartblock(self, imgobj):
        """
            获得验证码图片四边的可疑block, 并制定逻辑筛选出其中的曲线block
        :param imgobj:
        :return:
        """
        def filterowblocks(rowblocks):
            if len(rowblocks) in [0, 1]:
                return []
            # 统计所有block长度的分布
            lendict = {}
            for block in rowblocks:
                b_len = (block[1] - block[0])       # 这里计算长度时不要加1, 因为前面计算block的时候已经考虑了(下同)
                if b_len not in lendict.keys():
                    lendict[b_len] = 1
                else:
                    lendict[b_len] += 1

            # 以block数量最多的长度为中心,1或2为半径的长度范围内的所有blcok保留,其余的删除
            pivot = max(lendict.keys())
            ansblocks = [b for b in rowblocks if (b[1]-b[0]) in [pivot-1, pivot, pivot+1]]
            return ansblocks

        def filterColblocks(firstcolblocks, lastcolblocks):

            # 如果和图片左右两边有任何一边没有交点,从实际验证码情况来看,都是比较靠上下两边的曲线,不用去除
            if len(firstcolblocks) == 0 and len(lastcolblocks) == 0:
                return []
            elif len(firstcolblocks):
                return [firstCol_blocks[0]]
            elif len(lastcolblocks):
                return [lastcolblocks[0]]
            # 统计第一列所有block长度对应的block
            lendict = {}
            for block in firstcolblocks:
                b_len = (block[1] - block[0])
                if b_len not in lendict.keys():
                    lendict[b_len] = [block]
                else:
                    lendict[b_len].append(block)

            # 从第最后一列中找出和第一列中长度最接近的一块作为曲线块
            for block in lastcolblocks:
                b_len = (block[1] - block[0])
                for l in [b_len - 1, b_len, b_len + 1]:
                    if l in lendict.keys():
                        return [lendict[l][0]]
            print "没有找到合适的左册起始block..."
            return [lendict[max(lendict.keys())][0]]      # 返回最大的那一块
            pass

        ansdic = {}
        # 第一行的曲线block
        firstRow_blocks = self.getsuspectblocks(0, "R")
        ansdic["top"] = filterowblocks(firstRow_blocks)

        # 最后一行的曲线block
        lastrow_blocks = self.getsuspectblocks(39, "R")
        ansdic["bottom"] = filterowblocks(lastrow_blocks)

        # 第一列\最后一列的曲线block
        firstCol_blocks = self.getsuspectblocks(0, "C")
        lastCol_blocks = self.getsuspectblocks(139, "C")
        ansdic["left"] = filterColblocks(firstCol_blocks, lastCol_blocks)
        return ansdic
        pass


    def crossarea(self, indexofRC):
        """ 用于判断扫描法中当前block是否是和字母或数字的重合部位 """

        pass

    def getnextblock(self, old_block, new_block_list, mode):
        """
             根据上一行(列)的block,编写逻辑(大小比较,计算距离), 筛选出new_block中的曲线block
        :param old_block:
        :param new_block:
        :param mode:
        :return:
        """
        def crossarea(block, imgobj, threshold=35):
            """
            统计以width为半径,以point为中心的矩形区域中黑点个数,设置阈值,小于阈值就抹除该点
                :param width: 曲线的宽度
                :param point: 当前待处理的点(是否抹除)
                :param imgobj: 图像矩阵
                :return: True:抹除,False:不抹除
            """
            h, w = imgobj.shape

            # r, c = point
            # # print width
            # rs = 0 if r - width < 0 else r - width
            # re = h - 1 if r + width > h - 1 else r + width
            # cs = 0 if c - width < 0 else c - width
            # ce = w - 1 if c + width > w - 1 else c + width
            # block[]
            # safesquare = imgobj[rs:re, cs:ce]
            # # threshold = int(4 * (width ** 2) * 0.5)
            # threshold = int(4 * (width ** 2) * 0.4)
            # return True if len(safesquare[safesquare < 200]) < threshold else False

        if mode == 'C':
            if len(new_block_list) > 2:
                return []
            oldlen = (old_block[1] - old_block[0])
            new_block_list = [block for block in new_block_list if (block[1]-block[0]) in range(oldlen-2, oldlen+2)]
            if new_block_list:
                dislist = [self.calcblockdis(old_block, new_block=b) for b in new_block_list]
                # if min(dislist) > oldlen:  # 距离太远的相似block不删
                #     return []
                return new_block_list[dislist.index(min(dislist))]
            else:
                return []
            pass
        elif mode == "R":
            pass
        else:
            return False
        pass

    def rmcurseblock(self, obj_block, indexofRC, mode):
        """ 将obj_block指定的区域抹除 """
        if obj_block:
            if mode == "C":
                self.imgobj[obj_block[0]:obj_block[1], indexofRC] = 255
            elif mode == "R":
                self.imgobj[indexofRC, obj_block[0]:obj_block[1]] = 255
        pass




    def colscan(self, s_block, img_obj):
        """
            对二值化验证码图片做列扫描
        :param s_block:
        :param img_block:
        :return:
        """
        h, w = img_obj.shape
        # print h, w
        startblock = s_block["left"]        # 左侧起始block
        # print startblock
        old_block = startblock[0]
        # print old_block[1] - old_block[0]
        for c in range(1, w-1):
            newblocks = self.getsuspectblocks(c, mode="C")          # 当前列的所有block
            newblock = self.getnextblock(old_block, newblocks, mode="C")    # 当前列的曲线block
            if newblock:
                self.rmcurseblock(newblock, c, mode="C")
                old_block = newblock

        pass


    def getcycles(self, blocklists, mode):
        """
            根据上边或下边相交的blocks确定周期T, 从而确定参数w(欧米伽)
            返回T,以及计算T的两个points: p1, p2
            mode in ["T"(top), "B"(Bottom)]
        """
        # def countlen(dislen, lendic):
        #     if dislen in lendic.keys():
        #         lendic[dislen] += 1
        #     else:
        #         lendic[dislen] = 1
        #
        # oldblock = blocklists[0]
        # distancedic = {}
        # for block in blocklists[1:]:
        #     dlen_1 = block[1] - oldblock[1]
        #     dlen_2 = block[0] - oldblock[0]
        #     countlen(dlen_1, distancedic)
        #     countlen(dlen_2, distancedic)
        if len(blocklists) < 3:
            return [], [], [],[]
        T = blocklists[2][0] - blocklists[0][0]
        print "周期: %s" % T
        if mode == "T":
            y1 = y2 = 0
        else:
            y1 = y2 = 44
        p1 = (blocklists[0][0], y1)
        p2 = (blocklists[2][0], y2)
        width = blocklists[2][1] - blocklists[0][1]
        return T, p1, p2, width
        pass

    def getclacpoints(self, p1):
        """
            根据给定的交点, 求解其相邻右侧一列的最近block的外侧点p3
            返回p3
        """
        # 如果曲线和左边有交点,取其中点,为p3
        leftblcok = self.s_blocks["left"]
        if leftblcok:
            leftblcok = leftblcok[0]
            p3 = (0, (leftblcok[1]-leftblcok[0])/2,)
            return p3
        else:
            return None
        # 否则, 根据mode类型, 往上或者往下扫描一行, 计算和上一行中指定的block距离最近的一个block, 并从中选择一个点
        # blocks = self.getsuspectblocks(p1[0]+1, "C")
        # bs = [block[0] for block in blocks]
        # nextpoint_r = max(bs)
        # p3 = (p1[1], nextpoint_r)
        # # K = float(p3[1]-p1[1])/float(p3[0]-p1[0])
        # return p3
        pass

    def isfit(self, parmaslist, p1, p2):
        """
        计算已知曲线在x1与x2之间的所有点与已知的点的之间的重合度, 阈值设置为0.9,
        :param parmaslist:  (w, fie, A, k,)
        :param p1: 求解周期的左侧点
        :param p2: 求解周期的右侧点
        :return: True or False, 超过阈值即返回True, 否则返回False
        """
        def sinfunc(paramslist, x):
            w, fie, A, k = paramslist
            return int(A*math.sin(w*x + fie) + k)

        hitsum = 0
        totalpixels = 0
        for c in range(p1[0], p2[0]):
            r = sinfunc(parmaslist, c)
            if r in range(0, 44):
                totalpixels += 1
                if self.imgobj[r, c] < 20:
                    hitsum += 1
                self.imgobj[r, c] = 0
        print "totalpixels: %s" % totalpixels
        print "hitpixels: %s" % hitsum
        hit_accuracy = 0 if not totalpixels else float(hitsum) / totalpixels
        print hit_accuracy
        return True if (totalpixels > 50 and hit_accuracy > 0.9) else False
        pass

    def calcparams(self, p1, p2, p3, T, width):
        """
            利用已知条件求解出所有参数(暴力破解N, 求得fie就行)
            :param p1:
            :param p2:
            :param K:
            :param T:
            :return:
        """
        def getA(p1, p3, fie, w):
            division = (math.sin(w*p3[0]+fie) - math.sin(w*p1[0]+fie))
            if division:
                return float(p3[1] - p1[1]) / division
            else:
                return 0
                # return float(K)/w*math.cos(fie + w*p3[0])

        def getk(A, w, fie, p3):
            return p3[1] - A*math.sin(w*p3[0]+fie)
            # return float(p3[1] - A*math.sin(w*p3[0]+fie))

        def getFie(N, w, p1, p2):
            return float(N*math.pi - (p1[0]+p2[0])*w)/2

        w = float(2 * math.pi / T)

        for N in range(-2, 3):
            fie = getFie(N, w, p1, p2)
            A = getA(p1, p3, fie, w)
            if not A:
                # print "next"
                continue
            k = getk(A, w, fie, p2)
            if self.isfit((w, fie, A, k), p1, p2):
                return (A, w, fie, k)
        return None
        pass

    def rmsinfunc(self, paramslist):
        def sinfunc(paramslist, x):
            w, fie, A, k = paramslist
            return int(A * math.sin(w * x + fie) + k)

        def setrmarea(r, c, width=3):
            # h, w = self.imgobj.shape
            # rs = 0 if r - width < 0 else r - width
            # re = h - 1 if r + width > h - 1 else r + width
            # cs = 0 if c - width < 0 else c - width
            # ce = w - 1 if c + width > w - 1 else c + width
            # self.imgobj[rs:re,cs:ce] = 0
            self.imgobj[r,c] = 0

        for c in range(0, 140):
            r = sinfunc(paramslist, c)
            if r in range(0, 44):
                setrmarea(r, c)
        pass

    def rowscan(self, s_block):
        """ 利用已知的曲线方程进行曲线消除 """
        if not s_block["bottom"] and not s_block["top"]:
            return False
        # 确定曲线和图片上或下边交接的blocks
        blocklist, mode = (s_block["bottom"], "B") if len(s_block["bottom"]) > len(s_block["top"]) else (s_block["top"], "T")

        T, p1, p2, width = self.getcycles(blocklist, mode=mode)
        if not T:
            return False
        p3 = self.getclacpoints(p1)
        if not p3:
            print 'p3 not found'
            return False
        params = self.calcparams(p1, p2, p3, T, width)
        if params:
            self.rmsinfunc(params)
            # print params
            return True
        else:
            print "did't got func"
            return False
        pass

    def run(self, b_threshold=100):
        """ """
        imgobj = self.tobinarypic(b_threshold=b_threshold)  # 这里的 b_threshold是二值化时的阈值,默认是100,
        # 这是根据微信搜狗验证码的特性(背景字母颜色淡)得到的
        # self.rmeightNeibornoisepoint()      # 八邻域降噪
        # s_block = self.getstartblock(imgobj)
        # # self.s_blocks = s_block
        # # self.rowscan(s_block=s_block)
        # # if not self.rowscan(s_block=s_block):
        # # print s_block
        # self.colscan(s_block, imgobj)
        # # savename = self.fname.replace('.jpg', '_c.jpg')
        savename = self.imgpath + '_c.jpg'
        cv2.imwrite(savename, imgobj)
        pass

class rmcurlinebycolor():

    def ehanceimg(self, imgobj, mode=0):
        """ 图片增强的几种尝试
        mode:
            0: equalize
            1: laplace
            2: logimgenhance
            3: gamma
        """
        # image = imgobj
        # image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        #  直方图均衡增强
        def equalize(image):
            # image_equal = cv2.equalizeHist(image_gray)
            r, g, b = cv2.split(image)
            r1 = cv2.equalizeHist(r)
            g1 = cv2.equalizeHist(g)
            b1 = cv2.equalizeHist(b)
            image_equal_clo = cv2.merge([r1, g1, b1])
            img_gray = cv2.cvtColor(image_equal_clo, cv2.COLOR_BGR2GRAY)
            # self.cvshowimg(img_gray)
            return image_equal_clo

        #  拉普拉斯算法增强
        def laplace(image):
            kernel = numpy.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            return image_lap

        #  对象算法增强
        def logimgenhance(image):
            image_log = numpy.uint8(numpy.log(numpy.array(image) + 1))
            cv2.normalize(image_log, image_log, 0, 255, cv2.NORM_MINMAX)
            #    转换成8bit图像显示
            cv2.convertScaleAbs(image_log, image_log)
            return image_log

        #  伽马变换
        def gamma(image):
            fgamma = 2
            image_gamma = numpy.uint8(numpy.power((numpy.array(image) / 255.0), fgamma) * 255.0)
            cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
            cv2.convertScaleAbs(image_gamma, image_gamma)
            return image_gamma

        funclist = (equalize, laplace, logimgenhance, gamma,)
        func = funclist[mode]
        imgobj = func(imgobj)
        # imgobj = equalize(imgobj)

        # cv2.imshow('res', imgobj)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return imgobj


    def getfrontground(self, imgobj):
        """ 获取验证码图片的前景图中的字母和曲线 """
        img1 = imgobj
        # img2 = copy.deepcopy(img1)
        rows, cols, channels = img1.shape
        # img2 = numpy.zeros((rows, cols, 3), dtype=numpy.uint8)
        # img2 = cv2.bitwise_not(img2)
        roi = img1[0:rows, 0:cols]
        # 灰度图、二值化、制作掩膜（mask_inv）
        img2gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # self.cvshowimg(mask_inv)
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
        img2_fg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # self.getpointslist(img2_fg)
        # self.cvshowimg(img2_fg)
        # 直方图均衡增强图片
        img2_fg = self.ehanceimg(img2_fg, 0)
        self.cvshowimg(img2_fg)
        # 高斯滤波
        # blur = cv2.GaussianBlur(img2_fg, (3, 3), 0)
        blur = cv2.bilateralFilter(img2_fg,11, 105, 105)
        self.cvshowimg(blur)
        # # 灰度图
        # img2_fg_gray = cv2.cvtColor(img2_fg, cv2.COLOR_BGR2GRAY)
        # self.cvshowimg(img2_fg_gray)
        # equ = cv2.equalizeHist(img2_fg_gray)
        # self.cvshowimg(equ)

        # 转到LAB空间
        # img2_fg_LAB = cv2.cvtColor(img2_fg, cv2.COLOR_BGR2LAB)
        # print img2_fg_LAB
        # self.cvshowimg(img2_fg_LAB)
        # img2_fg = self.ehanceimg(img2_fg_LAB, 0)
        # self.cvshowimg(img2_fg)
        # 中值滤波
        # img2_fg_median = cv2.medianBlur(img2_fg, 3)  # 中值滤波:效果最好，但是会连带删除部分字体
        # self.cvshowimg(img2_fg_median)
        # 转换到HSV空间
        # img2_fg = cv2.cvtColor(img2_fg, cv2.COLOR_BGR2HSV)
        # self.cvshowimg(img2_fg)
        # dst = cv2.add(img2_fg, img2)
        # img1[0:rows, 0:cols] = dst
        # cv2.imshow('res', img2_fg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print img2_fg
        # img2_fg = self.ehanceimg(img2_fg, 0)

        # cv2.imshow('res', img2_fg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img2_fg

    def rmeightNeibornoisepoint(self, imgobj):
        """ 用8领域法去除噪声点,特别是连续曲线中的噪声点 """
        def pixel_8_neibor(r, c, imgobj):
            black_count = 0
            if r-1 >= 0:
                black_count += 1 if imgobj[r-1,c] < 200 else 0  # 左上方
                if c-1>=0:
                    black_count += 1 if imgobj[r-1,c-1] < 200 else 0  # 正上方
                if c+1<w:
                    black_count += 1 if imgobj[r-1, c+1] < 200 else 0  # 右上方
            if r+1 < h:
                black_count += 1 if imgobj[r+1, c] < 200 else 0    # 正下方
                if c-1>=0:
                    black_count += 1 if imgobj[r+1, c-1] < 200 else 0  # 左下方
                if c+1<w:
                    black_count += 1 if imgobj[r+1, c+1] < 200 else 0  # 右下方

            if c-1 >= 0:
                black_count += 1 if imgobj[r, c-1] < 200 else 0  # 左侧 
            if c+1 < w:
                black_count += 1 if imgobj[r, c+1] < 200 else 0  # 右侧

            #print black_count
            if black_count >= 4:
                imgobj[r, c] = 0
            else:
                print imgobj[r,c], black_count
                imgobj[r, c] = 255


        h, w = imgobj.shape
        print h, w
        for r in range(h):
            for c in range(w):
                # if imgobj[r,c] >= 200:
                pixel_8_neibor(r, c, imgobj)

    def turnimg2binary(self, imgobj):
        img2gray = cv2.cvtColor(imgobj, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)
        return mask

    def corrosion(self, imgobj=None, origin_frontimg=None, mode=1):
        """ 使用腐蚀操作，对细曲线进行去除"""
        # imgobj = cv2.imread("/home/jingdata/Document/LAB_CODE/captcha/Captcha_Recg/sogoucapture/9")

        if mode == 1:
            bk_imgobj = copy.deepcopy(imgobj)
        elif mode != 2 and type(origin_frontimg) == numpy.ndarray:
            bk_imgobj = copy.deepcopy(origin_frontimg)
        else:
            print "corrosion函数参数输入有误!"
            return imgobj
        imgobj = cv2.cvtColor(imgobj, cv2.COLOR_BGR2GRAY)
        ret, imgobj = cv2.threshold(imgobj, 100, 255, cv2.THRESH_BINARY)
        old_th = copy.deepcopy(imgobj)
        # imgobj = cv2.bitwise_not(imgobj)
        # kernel = numpy.ones((1,1), numpy.uint8)
        # imgobj = cv2.erode(imgobj, kernel, iterations = 1) # 腐蚀
        # imgobj = cv2.bitwise_not(imgobj)
        # cv2.imwrite("/home/jingdata/Document/LAB_CODE/corrosion.jpg", imgobj)
        # imgobj = cv2.dilate(imgobj ,kernel,iterations = 1)	 # 膨胀
        # self.rmeightNeibornoisepoint(imgobj)			# 八领域降噪
        # imgobj = cv2.bitwise_not(imgobj)
        # imgobj = cv2.Canny(imgobj, 100, 200)  # 找边界
        imgobj = cv2.bitwise_not(imgobj)
        contours, hierarchy = cv2.findContours(imgobj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓只能针对黑底白字,并且会对原图操作
        white_plank = numpy.zeros((44, 140, 3), dtype=numpy.uint8)
        # white_plank = cv2.bitwise_not(white_plank)
        white_plank[:] = [255, 255, 255]
        # print white_plank.shape, imgobj.shape
        # print contours
        # print "type(contours):%s" % type(contours[1]), contours[1]
        # contours.sort(cmp=cv2.contourArea, reverse=True)
        contours = [C for C in contours if cv2.contourArea(C) > 5]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        print len(contours)
        legth = len(contours) if len(contours) < 10 else 10
        for i in range(0, legth):
            area = cv2.contourArea(contours[i])
            # print area
            # if area < 50:
            #    continue
            x, y, w, h = cv2.boundingRect(contours[i])
            # print x, y, w, h
            # white_plank[y:y + h, x:x + w] = bk_imgobj[y:y + h, x:x + w]
            cv2.rectangle(old_th, (x, y), (x+w, y+h), (0, 0, 255), 1)

        cv2.drawContours(white_plank, contours, -1, (0, 0, 0), -1)
        white_plank = self.turnimg2binary(white_plank)
        mask = cv2.bitwise_not(white_plank)
        final_img = cv2.bitwise_and(bk_imgobj, bk_imgobj, mask=mask)
        self.getpointslist(final_img)
        # self.cvshowimg(old_th)
        # self.cvshowimg(white_plank)
        # self.cvshowimg(final_img)
        return final_img
        # print contours
        # imgobj = cv2.bitwise_not(imgobj)
        # cnt = contours[0]
        # cv2.drawContours(imgobj, contours[1], -1, (0,0,255), 1)
        # x,y,w,h = cv2.boundingRect(cnt)
        #print x,y,w,h
        #cv2.rectangle(imgobj, (x, y), (x+w, y+h),(0,0,255), 2)

        # imgobj = cv2.bitwise_not(imgobj)
        # cv2.imshow('res', old_th)	# 如果显示的图片太小，可以使用cv2.resize()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    pass

    def findcontours(self):
        """ 寻找轮廓 """
        pass


    def getdistlist_from_frontground(self, imgobj, discalc_mode='color'):
        """
            用mask获得img的前景图，并且从中提取出所有的前景点，计算任意两个点之间的距离并返回
        返回值：
            pointslist: 包含所有前景点坐标的列表
            distlist: 距离列表，用于聚类时构造距离矩阵，保存的值为pointslist中任意两点之前的距离
            imgobj: mask之后的前景图
        """
        imgobj = self.getfrontground(imgobj)
        # print imgobj[0,:]
        pointslist = self.getpointslist(imgobj)
        # cv2.imshow('res', imgobj)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        distlist = self.calspaceORcolordist(pointslist, mode=discalc_mode)      # discalc_mode取值可以是color, s_c, space
        pointslist = [(p[0], p[1]) for p in pointslist]
        return pointslist, distlist, imgobj
        pass

    def rmobjbycolor(self):
        """ 对原始图片从像素值角度进行统计：设置阈值，直接对原图片进行操作，对于在指定阈值范围内的像素点进行抹除 """
        files = os.listdir("sogoucapture/funcCruse")
        abpath = os.path.abspath("sogoucapture/funcCruse")
        rgbdict = {}
        for f in files[0:1]:
            print "start processing file %s......" % f
            fname = os.path.join(abpath, f)
            if not os.path.isdir(fname):
                imgobj = cv2.imread(fname)
                h, w = (44, 140)
                imgobj = self.getfrontground(imgobj)
                # print imgobj[0,:]
                pointslist = self.getpointslist(imgobj)
                distlist = self.calspaceORcolordist(pointslist, mode='color')  # mode取值可以是color, s_c, space
                pointslist = [(p[0], p[1]) for p in pointslist]
                return pointslist, distlist, imgobj
                # break
                #     imgobj[imgobj==0] = 255
                #     for r in range(0, h):
                #         for c in range(0, w):
                #             tkey = "_".join("{0}".format(n) for n in imgobj[r, c])
                #             if tkey == "255_255_255":
                #                 continue
                #             if tkey in rgbdict.keys():
                #                 rgbdict[tkey] += 1
                #             else:
                #                 rgbdict[tkey] = 1
                #     maxv = 0
                #     maxk = ""
                #     for k, v in rgbdict.items():
                #         print k, v
                #         if v > maxv:
                #             maxk = k
                #             maxv = v
                #         if v == 48:
                #             print k
                #
                #     vlist = [int(i) for i in maxk.split("_")]
                #
                #     for r in range(0, h):
                #         for c in range(0, w):
                #             # print vlist
                #             if list(imgobj[r, c]) == [165, 173, 166]:
                #                 # print imgobj[r,c]
                #                 imgobj[r, c] = [255, 255, 255]
                #                 # print imgobj[r,c]
                #     cv2.imwrite("%s_c.jpg" % fname, imgobj)
                # print "end processing file %s......\n" % f

    def getpointslist(self, imgobj):
        """ mask后的图片中提取对应的点
        返回pointlist=[(x, y, [*,*,*]), ...]
        """
        #imgobj[]
        # print type(imgobj)
        # imgobj = cv2.bitwise_not(imgobj)
        pointslist = []
        h, w, channel = imgobj.shape
        for r in range(h):
            for c in range(w):
                c_list = list(imgobj[r, c])
                if c_list != [0, 0, 0] and c_list != [255, 255, 255]:     # 不是白色或者黑色:不是背景色 LAB空间:and c_list != [0, 128, 128]
                    pointslist.append([r, c, imgobj[r, c]])
                elif c_list == [0, 0, 0] :						# 这边可能存在问题，如果字符或者曲线是黑色先画出来的，就有可能误删  # LAB空间:or c_list == [0, 128, 128]
                    imgobj[r, c] = [255, 255, 255]
                    pass
        # print pointslist
        # cv2.imshow('res', imgobj)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return pointslist


    def calspaceORcolordist(self, pointslist, mode='color'):
        """ 计算任意两点之间的距离：二维实体空间的距离、颜色空间的距离或者二者的综合 """

        def colordist(p1, p2):
            """ 计算颜色空间的距离 """
            p1_c = p1[2]
            p2_c = p2[2]
            cdist = numpy.sqrt(numpy.sum(numpy.square(p1_c - p2_c)))
            return cdist

        def HScolordist(p1, p2):
            p1_c = p1[2][:2]
            p2_c = p2[2][:2]
            cdist = numpy.sqrt(numpy.sum(numpy.square(p1_c - p2_c)))
            return cdist

        def colordist2(p1,p2):
            """ 改进的加权欧氏距离"""
            p1_c = p1[2]
            p2_c = p2[2]
            r = ((p1_c[0] + p2_c[0]) / 2.0)
            delta_r = numpy.square(p1_c[0] - p2_c[0])
            delta_g = numpy.square(p1_c[1] - p2_c[1])
            delta_b = numpy.square(p1_c[2] - p2_c[2])
            # print r, delta_b, type(r), type(delta_b)
            cdist = numpy.sqrt(numpy.sum((2 + r/256.0) * delta_r + 4 * delta_g + (2+(255.0-r)/256.0) * delta_b))
            return cdist
            pass

        def spacedist(p1, p2):
            """ 计算二维实体空间的距离 """
            p1_s = numpy.array(p1[:2])
            p2_s = numpy.array(p2[:2])
            sdist = numpy.sqrt(numpy.sum(numpy.square(p1_s - p2_s)))
            return sdist


        def fullmatrix(distlist, totalpoints):
            matrixlist = []
            n = totalpoints
            # 将右上三角矩阵扩充成对称矩阵
            for i in range(n):
                rowlist = [0] * (i+1)
                s_index = i*n - (1+i)*i / 2
                e_index = (i+1)*n - (i+2)*(i+1)/2
                rowlist.extend(distlist[s_index:e_index])
                matrixlist.append(rowlist)
            M = numpy.array(matrixlist)
            M += M.T - numpy.diag(M.diagonal())
            # 美化输出
            import pprint
            pprint.pprint(M)
            return M
            pass

        def combinespacedist(p1, p2):
            """ 将颜色空间和距离空间合并成五维空间进行距离计算 """
            # 归一化
            # p1[0] /= 44.0
            # p1[1] /= 140.0
            # p1[2] = p1[2] / 255.0
            # p2[0] /= 44.0
            # p2[1] /= 140.0
            # p2[2] = p2[2] / 255.0
            # 空间扩展
            p1_sc = numpy.concatenate([p1[:2], p1[2]])
            p2_sc = numpy.concatenate([p2[:2], p2[2]])
            sc_dist = numpy.sqrt(numpy.sum(numpy.square(p1_sc, p2_sc)))
            return sc_dist
            pass

        def space_color_dist(pointslist):
            """ 计算实体_颜色空间的距离: 归一化后，各自权重取0.5 """
            # 分别计算颜色空间距离和实体空间距离
            clrdist = []
            spcdist = []
            sc_dist = []
            for i, fp in enumerate(pointslist):
                for ep in pointslist[i + 1:]:
                    clrdist.append(colordist(fp, ep))
                    spcdist.append(spacedist(fp, ep))
                    # sc_dist.append(combinespacedist(fp, ep))      # 空间合并:失败

            # # 将上述上三角矩阵扩充成完整的对角矩阵，并对每行求和
            # c_M = fullmatrix(clrdist, len(pointslist))  # 将上述上三角矩阵扩充成完整的对角矩阵
            # lin_c_sum = map(sum, c_M)  # 对颜色空间距离矩阵的每行求和
            # s_M = fullmatrix(spcdist, len(pointslist))  # 将上述上三角矩阵扩充成完整的对角矩阵
            # lin_s_sum = map(sum, s_M)  # 对实体空间距离矩阵的每行求和
            #
            # # 对距离列表中的元素进行归一化操作
            # c = 0
            # for i, _ in enumerate(pointslist):
            #     for j, _ in enumerate(pointslist[i + 1:]):
            #         clrdist[c] /= float(lin_c_sum[i])
            #         spcdist[c] /= float(lin_s_sum[i])
            #         c += 1

            # 对两个距离列表分别进行求和，并分别进行归一化
            c_sum = float(sum(clrdist))
            s_sum = float(sum(spcdist))
            print c_sum
            print s_sum
            clrdist = [c/c_sum for c in clrdist]
            spcdist = [s/s_sum for s in spcdist]

            s_c_dist = [0.8 * c + 0.2 * s for c, s in zip(clrdist, spcdist)]  # # 按照各0.5的权值叠加颜色距离和实体距离:
            # s_c_dist = [math.sqrt(c**2+s**2) for c, s in zip(clrdist, spcdist)]  # 改进,将(颜色距离,实体距离) 作为新的坐标,计算欧式距离
            # s_c_dist = [c*s for c, s in zip(clrdist, spcdist)]
            return s_c_dist
            pass

        # 如果是计算颜色_实体空间距离
        if mode == "s_c":
            return space_color_dist(pointslist)

        # 否则，再判断是计算颜色空间距离或实体空间距离
        distlist = []
        for i, fp in enumerate(pointslist):
            for ep in pointslist[i+1:]:     # 这里做切片的时候，如果i+1 == len(pointslist), 则切片返回[],所以这样写是安全的
                if mode == 'color':
                    distlist.append(colordist(fp, ep))
                    # distlist.append(HScolordist(fp, ep))
                else:
                    distlist.append(spacedist(fp, ep))
                pass
        # print distlist
        return distlist

    def drawpic(self, pointslist, labels):
        """ """
        f2 = plt.figure(1)
        ax = f2.add_subplot(111)
        colorlsit = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'bc', 'gk', 'mr', 'ybc', 'bcg']
        picpointsdic = {}
        for p, l in zip(pointslist, labels):
            if l in picpointsdic.keys():
                picpointsdic[l].append(tuple(p))
            else:
                picpointsdic[l] = [tuple(p)]
        for k, v in picpointsdic.items():
            x = [i[1] for i in v]
            y = [-j[0] for j in v]
            # print x, y
            ax.scatter(x, y, color=colorlsit[k], label=k, s=10)
        ax.legend(loc="upper right")
        plt.show()
        pass

    def getbiggestypepointset(self, pointslist, labels):
        """ 讲聚类后的每一类点投影到x轴，覆盖面积最大的默认是曲线 """
        ansdic = {}
        for p, l in zip(pointslist, labels):
            if l in ansdic.keys():
                ansdic[l].append(tuple(p))
            else:
                ansdic[l] = [tuple(p)]
        max = 0
        maxk = 0
        for k, v in ansdic.items():
            v = [i[1] for i in v]   # 投影到x轴
            if len(set(v)) > max:
                max = len(set(v))
                maxk = k
        # print maxk
        bigslabes = [label for label in labels if label == maxk]
        return ansdic[maxk], bigslabes, ansdic
        pass

    def rmcurselinepoints(self, imgobj, pointslist):
        # print imgobj
        for p in pointslist:
            imgobj[p[0], p[1]] = [255, 255, 255]
        pass

    def rmCurseline_by_Agglocluster(self, imgobj, discalc_mode='color'):
        """利用层次聚类算法对mask后的图片进行曲线和字母分离"""
        # pointslist, distlist, imgobj = self.rmobjbycolor()
        # 数据准备：计算前景点和距离列表
        # 其中参数discalc_mode取值可以是color, s_c, space, 分别对应颜色聚类、颜色_实体空间聚类、实体空间聚类
        pointslist, distlist, imgobj = self.getdistlist_from_frontground(imgobj, discalc_mode=discalc_mode)
        bk_imgobj = copy.deepcopy(imgobj)
        # 层次聚类
        model = AgglomerativeClustering(n_clusters=7, affinity='precomputed', linkage='average')
        dist_matrix = scidst.squareform(numpy.array(distlist))      # 根据距离列表构造距离矩阵
        labels = model.fit_predict(dist_matrix)
        # metrics.silhouette_score(dist_matrix, labels=labels, metric="precomputed") # 轮廓系数，检测聚类质量
        # print len(pointslist), len(labels)

        # 根据前景点数据和对应的label画图
        self.drawpic(pointslist, labels)

        # 找出曲线所在的类（通过投影的方式），将该类的点全部置成背景色
        # curselinepoints, biggestLabels, lable_pdict = self.getbiggestypepointset(pointslist, labels)
        # curselinepoints = self.freshkmeanscurselinepoints(cursepoints=curselinepoints, kmpointsets_dict=lable_pdict, bk_imgobj=imgobj)
        # self.rmcurselinepoints(imgobj, curselinepoints)
        # self.cvshowimg(imgobj)

        # self.drawpic(curselinepoints, biggestLabels)
        # # self.cvshowimg(imgobj)
        # # kernel = numpy.ones((3,3), numpy.uint8)   #
        # # imgobj = cv2.erode(imgobj, kernel, iterations = 1) # 腐蚀
        # # imgobj = cv2.dilate(imgobj ,kernel,iterations = 1)	 # 膨胀
        # # imgobj = cv2.GaussianBlur(imgobj, (3,3), 0) # 高斯滤波
        # # imgobj = cv2.medianBlur(imgobj, 3)    # 中值滤波:效果最好，但是会连带删除部分字体
        # # imgobj = cv2.blur(imgobj, (3, 3))     # 平均:效果差
        #
        # img2gray = cv2.cvtColor(imgobj, cv2.COLOR_BGR2GRAY)
        # ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)
        # mask = cv2.bitwise_not(mask)
        # kernel = numpy.ones((3, 3), numpy.uint8)
        # mask_inv = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # self.cvshowimg(mask_inv)
        # imgobj = cv2.bitwise_and(bk_imgobj, bk_imgobj, mask=mask_inv)
        # self.getpointslist(imgobj)  # 只是为了将黑色背景置成白色, 并不是为了获得前景图的像素点
        # self.cvshowimg(imgobj)
        # imgobj = self.corrosion(imgobj)  #  origin_frontimg=bk_imgobj, mode=2
        # self.cvshowimg(imgobj)
        # # cv2.namedWindow('res', 0)
        # # imgobj = cv2.medianBlur(imgobj, 3)    # 中值滤波:效果最好，但是会连带删除部分字体
        # self.cvshowimg(imgobj)
        pass


    def rmCurseline_by_DBSCANCluster(self, imgobj, discalc_mode='color'):
        # 数据准备：计算前景点和距离列表
        # 其中参数discalc_mode取值可以是color, s_c, space, 分别对应颜色聚类、颜色_实体空间聚类、实体空间聚类
        pointslist, distlist, imgobj = self.getdistlist_from_frontground(imgobj, discalc_mode=discalc_mode)

        # 层次聚类
        dist_matrix = scidst.squareform(numpy.array(distlist))  # 根据距离列表构造距离矩阵
        db = DBSCAN(eps=5, metric='precomputed', min_samples=20)
        labels = db.fit_predict(dist_matrix)
        # print labels
        # print len(pointslist), len(labels)

        # 根据前景点数据和对应的label画图
        self.drawpic(pointslist, labels)

        # 找出曲线所在的类（通过投影的方式），将该类的点全部置成背景色
        curselinepoints, biggestLabels, lable_pdict = self.getbiggestypepointset(pointslist, labels)
        # self.drawpic(curselinepoints, biggestLabels)
        # self.rmcurselinepoints(imgobj, curselinepoints)
        pass


    def cvshowimg(self, imgobj):
        cv2.imshow('res', imgobj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.waitKey(1)

    def freshkmeanscurselinepoints(self, cursepoints, kmpointsets_dict, bk_imgobj):
        """  对kmeans聚类的结果进行修正,找到最合适的点集合 """
        def averageColordist(crspoints, suspectpointset):
            """ 用来计算任意两个点集的平均距离 """
            dist_dic = {}
            setsdist = 0.0
            for p in suspectpointset:
                k = "%s_%s" % p
                p_sum = 0.0
                for cp in crspoints:
                    p_c = bk_imgobj[p]
                    cp_c = bk_imgobj[cp]
                    p_sum += numpy.sqrt(numpy.sum(numpy.square(p_c - cp_c)))
                dist_dic[k] = p_sum / len(crspoints)
                setsdist += dist_dic[k]
            setsdist /= len(suspectpointset)
            return dist_dic, setsdist

        mindist = 900000
        nearest_label = 0
        near_dis_dic = {}
        for k, v in kmpointsets_dict.items():
            dis_dic, setsdist = averageColordist(cursepoints, v)
            if setsdist < mindist:
                nearest_label = k
                near_dis_dic = dis_dic
                mindist = setsdist


        # 对集合中的噪声点进行删除
        winsets_plist = kmpointsets_dict[nearest_label]
        for k, v in near_dis_dic.items():
            if v > 1 * mindist:
                noise_p = tuple([int(noip) for noip in k.strip().split("_")])
                print noise_p
                winsets_plist.remove(noise_p)
        return winsets_plist
        # return kmpointsets_dict[k], near_dis_dic
        pass


    def rmCurseline_by_Kmeans(self, imgobj, discalc_mode='color'):
        """ 利用密度聚类算法对mask后的图片进行曲线和字母分离"""
        # 数据准备：计算前景点和距离列表
        # 其中参数discalc_mode取值可以是color, s_c, space, 分别对应颜色聚类、颜色_实体空间聚类、实体空间聚类
        bk_imgobj = self.getfrontground(imgobj)
        # self.getpointslist(bk_imgobj)
        # img_tmp = copy.deepcopy(bk_imgobj)
        # first_cursepoints = []
        # for it in range(3):
        #     # print imgobj[0,:]
        #     pointslist = self.getpointslist(img_tmp)
        #     plist = [p[:2] for p in pointslist]     # 实体坐标
        #     clist = [p[2] for p in pointslist]      # 颜色坐标
        #
        #     datalist = []
        #     n_cluster = 7
        #     if discalc_mode == 'color':
        #         datalist = clist
        #         n_cluster = 10
        #     elif discalc_mode == 'space':
        #         datalist = plist
        #     estimator = KMeans(n_clusters=n_cluster+it)
        #     labels = estimator.fit_predict(datalist)
        #
        #     # print labels
        #     # print len(pointslist), len(labels)
        #
        #     # 根据前景点数据和对应的label画图
        #     # self.drawpic(plist, labels)
        #
        #     # 找出曲线所在的类（通过投影的方式），将该类的点全部置成背景色
        #     curselinepoints, biggestLabels, label_pdict = self.getbiggestypepointset(plist, labels)
        #     if it == 0:  # 保存最核心的curseline点集合
        #         first_cursepoints = curselinepoints
        #     if it > 10:   # 不能再用投影,要重新计算可疑曲线的点集合
        #         curselinepoints = self.freshkmeanscurselinepoints(first_cursepoints, label_pdict, bk_imgobj)
        #
        #     # 删除曲线点集合
        #     self.rmcurselinepoints(img_tmp, curselinepoints)
        #
        #     # 闭运算:先膨胀再腐蚀,这是为了重描被误删的字符部分
        #     if it == 2:  # not (it+1) % 2
        #         # print it
        #         # img_tmp = cv2.medianBlur(img_tmp, 3)  # 中值滤波:效果最好，但是会连带删除部分字体
        #         # img_tmp = self.corrosion(img_tmp)
        #         self.cvshowimg(img_tmp)
        #         img2gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
        #         ret, mask = cv2.threshold(img2gray, 100, 255, cv2.THRESH_BINARY)
        #         mask = cv2.bitwise_not(mask)
        #         kernel = numpy.ones((3, 3), numpy.uint8)
        #         mask_inv = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        #         img_tmp = cv2.bitwise_and(bk_imgobj, bk_imgobj, mask=mask_inv)
        #         self.getpointslist(img_tmp)  # 只是为了将黑色背景置成白色, 并不是为了获得前景图的像素点
        #         # img_tmp = cv2.medianBlur(img_tmp, 3)  # 中值滤波:效果最好，但是会连带删除部分字体
        #         self.cvshowimg(img_tmp)
        #     # #
        #     # if it == 2:
        #     #     img_tmp = self.corrosion(imgobj=img_tmp, origin_frontimg=bk_imgobj)
        #
        # img_tmp = self.corrosion(imgobj=img_tmp)
        # img_tmp = self.turnimg2binary(img_tmp)
        # self.cvshowimg(img_tmp)
        # # img_tmp = cv2.medianBlur(img_tmp, 3)  # 中值滤波:效果最好，但是会连带删除部分字体
        # # self.cvshowimg(img_tmp)
        # return img_tmp
        pass


    def RecogText(self, imgobj):
        """ 识别字符"""
        Img_new = Image.fromarray(imgobj)
        text = pytesseract.image_to_string(Img_new, lang='eng')
        print text

    def run(self):
        dirname = "sogoucapture/classicCaptcha"
        files = os.listdir(dirname)
        abpath = os.path.abspath(dirname)
        rgbdict = {}
        for f in files[0:]:
            print "start processing file %s......" % f
            fname = os.path.join(abpath, f)
            if not os.path.isdir(fname):
                imgobj = cv2.imread(fname)
                self.cvshowimg(imgobj)
                self.rmCurseline_by_Agglocluster(imgobj, discalc_mode='color')
                # self.rmCurseline_by_Agglocluster(imgobj, discalc_mode='space')
                # self.rmCurseline_by_DBSCANCluster(imgobj=imgobj, discalc_mode='s_c')
                # imgobj = self.rmCurseline_by_Kmeans(imgobj=imgobj, discalc_mode='color')

                # imgobj = self.rmCurseline_by_Kmeans(imgobj=imgobj, discalc_mode='space')
                # self.rmCurseline_by_Agglocluster(imgobj, discalc_mode='space')
                # self.rmCurseline_by_DBSCANCluster(imgobj=imgobj, discalc_mode='space')
                # self.cvshowimg(imgobj)
                # self.RecogText(imgobj)
                # cv2.imwrite(os.path.join(abpath, f+"_rn.jpg"), imgobj)
        pass

    def littlejoke(self, imgpath = "0802_1.jpg"):
        imgobj = cv2.imread(imgpath)
        h, w, channel = imgobj.shape
        imgobj = imgobj[550:560, 300:450]
        for _ in range(1):
            imgobj = self.ehanceimg(imgobj, 3)
            imgobj = self.ehanceimg(imgobj)
        self.cvshowimg(imgobj)
        pass

if __name__ == '__main__':
    # p1 = Image.open('sogoucapture/0') #
    # # text = pytesseract.image_to_string(p1, lang='chi_sim')
    # bimg = p1.convert('L')
    # bimg = bimg.convert('1')
    # bimg.save("testbinary.jpg")
    # bimg.show()  # sudo apt-get install imagemagick :如果不显示图片的话,安装一下


    # 使用联通量算法清除曲线
    tobj = rmBKnoise()
    # tobj.rmbkalpahb("sogoucapture/funcCruse/5")     # "captcha/AABW.jpg"
    # tobj.rmcurselines()
    # startblock, imgobj = tobj.getstartendblock("bwtest.jpg")
    # print startblock, imgobj
    # tobj.detectnextpixel(startblock, imgobj)
    # tobj.run()


    # 使用扫描算法实现曲线的清除
    #files = os.listdir("sogoucapture/")  # /funcCruse
    #abpath = os.path.abspath("sogoucapture/")  # /funcCruse
    #for f in files:
    #    print "start processing file %s......" % f
    #    fname = os.path.join(abpath, f)
    #    if not os.path.isdir(fname):
    #        tscanobj = rmcurselinewithscanning(fname)
    #        tscanobj.run()
    #    print "end processing file %s......\n" % f
    #pass


    # 从颜色角度去除曲线
    robj = rmcurlinebycolor()
    # robj.rmobjbycolor()
    #robj.corrosion() # 腐蚀、膨胀、找轮廓的尝试
    # robj.rmCurseline_by_Agglocluster()
    robj.run()
    # robj.littlejoke()


    # 列出opencv颜色转换的所有颜色空间
    # flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    # from pprint import pprint
    # pprint(flags)
    pass






    """
    颜色空间:
         'COLOR_BGR2BGR555',
         'COLOR_BGR2BGR565',
         'COLOR_BGR2BGRA',      # 3通道转到4通道
         'COLOR_BGR2GRAY',      # 转成灰度图
         'COLOR_BGR2HLS',       
         'COLOR_BGR2HLS_FULL',
         'COLOR_BGR2HSV',
         'COLOR_BGR2HSV_FULL',
         'COLOR_BGR2LAB',
         'COLOR_BGR2LUV',
         'COLOR_BGR2RGB',       # 改变3通道的通道顺序
         'COLOR_BGR2RGBA',      # 转成4通道,并且改变通道顺序
         'COLOR_BGR2XYZ',
         'COLOR_BGR2YCR_CB',
         'COLOR_BGR2YUV',
         'COLOR_BGR2YUV_I420',
         'COLOR_BGR2YUV_IYUV',
         'COLOR_BGR2YUV_YV12',
    """
