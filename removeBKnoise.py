#!/usr/bin/python
# -*- coding: utf-8 -*-

# Created by David Teng on 18-7-4


from PIL import Image
# import pytesseract
import cv2
import numpy
import Queue
import os
import math
import copy

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
        self.rmeightNeibornoisepoint()      # 八邻域降噪
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
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
        img2_fg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # dst = cv2.add(img2_fg, img2)
        # img1[0:rows, 0:cols] = dst
        return img2_fg
        # cv2.imshow('res', img2_fg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

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
                imgobj[imgobj==0] = 255
                for r in range(0, h):
                    for c in range(0, w):
                        tkey = "_".join("{0}".format(n) for n in imgobj[r, c])
                        if tkey == "255_255_255":
                            continue
                        if tkey in rgbdict.keys():
                            rgbdict[tkey] += 1
                        else:
                            rgbdict[tkey] = 1
                maxv = 0
                maxk = ""
                for k, v in rgbdict.items():
                    print k, v
                    if v > maxv:
                        maxk = k
                        maxv = v
                    if v == 48:
                        print k

                vlist = [int(i) for i in maxk.split("_")]

                for r in range(0, h):
                    for c in range(0, w):
                        # print vlist
                        if list(imgobj[r, c]) == [165, 173, 166]:
                            # print imgobj[r,c]
                            imgobj[r, c] = [255, 255, 255]
                            # print imgobj[r,c]
                cv2.imwrite("%s_c.jpg" % fname, imgobj)
            print "end processing file %s......\n" % f
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
    # tobj.rmbkalpahb("sogoucapture/0")     # "captcha/AABW.jpg"
    # tobj.rmcurselines()
    # startblock, imgobj = tobj.getstartendblock("bwtest.jpg")
    # print startblock, imgobj
    # tobj.detectnextpixel(startblock, imgobj)
    # tobj.run()


    # 使用扫描算法实现曲线的清除
    # files = os.listdir("sogoucapture")  # /funcCruse
    # abpath = os.path.abspath("sogoucapture")  # /funcCruse
    # for f in files:
    #     print "start processing file %s......" % f
    #     fname = os.path.join(abpath, f)
    #     if not os.path.isdir(fname):
    #         tscanobj = rmcurselinewithscanning(fname)
    #         tscanobj.run()
    #     print "end processing file %s......\n" % f
    # pass


    # 从颜色角度去除曲线
    robj = rmcurlinebycolor()
    robj.rmobjbycolor()



    pass