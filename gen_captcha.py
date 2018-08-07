#coding:utf-8
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
import random,time,os
import string
import json
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import math


class gencaptcha_cnn_98():
    # 验证码中的字符, 就不用汉字了
    number = ['0','1','2','3','4','5','6','7','8','9']
    alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    ALPHABET = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

    # 验证码一般都无视大小写；验证码长度4个字符
    def random_captcha_text(self, char_set=number+alphabet+ALPHABET, captcha_size=4):
        captcha_text = []
        for i in range(captcha_size):
            c = random.choice(char_set)
            captcha_text.append(c)
        return captcha_text

    # 生成字符对应的验证码
    def gen_captcha_text_and_image(self):
        image = ImageCaptcha()

        captcha_text = self.random_captcha_text()
        captcha_text = ''.join(captcha_text)

        captcha = image.generate(captcha_text)
        #image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

        #rm  =  'rm '+captcha_text + '.jpg'
        #print rm
        #os.system(rm)
        #time.sleep(10)

        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        return captcha_text, captcha_image

class gencapthca_test():


    # 字体的位置，不同版本的系统会有不同
    font_path = '/usr/share/fonts/truetype/fonts-japanese-gothic.ttf'
    # 生成几位数的验证码
    number = 6
    # 生成验证码图片的宽度和高度
    size = (140, 44)
    # 背景颜色，默认为白色
    bgcolor = (255, 255, 255)
    # 字体颜色，默认为蓝色
    fontcolor = (0, 0, 0)
    # 干扰线颜色。默认为红色
    linecolor = (0, 0, 0)
    # 是否要加入干扰线
    draw_line = True
    # 加入干扰线条数的上下限
    line_number = (1, 5)

    # 用来随机生成一个字符串
    def gene_text(self):
        # print string.letters
        source = list(string.letters)
        for index in range(0, 10):
            source.append(str(index))
        tmpletters = random.sample(source, self.number)
        print ''.join(tmpletters)
        return ''.join(tmpletters)  # number是生成验证码的位数

    # 用来绘制干扰线
    def gene_line(self, draw, width, height):
        begin = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([begin, end], fill=self.linecolor)

    # 生成验证码
    def gene_code(self):
        width, height = self.size  # 宽和高
        image = Image.new('RGBA', (width, height), self.bgcolor)  # 创建图片
        font = ImageFont.truetype(self.font_path, 25)  # 验证码的字体
        draw = ImageDraw.Draw(image)  # 创建画笔
        text = self.gene_text()  # 生成字符串
        font_width, font_height = font.getsize(text)
        print "font_width:%s, font_height:%s" % (font_width, font_height)
        draw.text(((width - font_width) / self.number, (height - font_height) / self.number), text,
                  font=font, fill=self.fontcolor)  # 填充字符串
        if self.draw_line:
            self.gene_line(draw, width, height)
        # image = image.transform((width+30,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
        # image = image.transform((width + 20, height + 10), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0),
        #                         Image.BILINEAR)  # 创建扭曲

        image = image.rotate(random.randint(-10, 10), expand=0)
        fff = Image.new('RGBA', image.size, (255,)*4)
        image = Image.composite(image, fff, image)
        image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强
        image.save('idencode.png')  # 保存验证码图片


class gencaptcha_final():
    def __init__(self, mode, totalnum, savepath="./captcha_set1", rotate=False, rangle=35, drawline=True):
        self.source = list(string.letters)
        for index in range(0, 10):
            self.source.append(str(index))
	stop_words = ['G', 'I', 'L', 'O', 'Q', 'U', 'W', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'q', 'r', 't', 'u', 'w', 'y', 'z', '0', '9']
	self.source = list(set(self.source)-set(stop_words)) 
        self.fontpool = [
            "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            "/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-B.ttf"
        ]
        # 总共需要生成的验证码的数量
        self.totalnum = totalnum
        # 验证码中包含的字符个数
        self.number = mode
        # 生成验证码图片的宽度和高度
        self.size = (140, 44)
        # 背景颜色，默认为白色
        self.bgcolor = (255, 255, 255)
        # 字体颜色，默认为蓝色
        self.fontcolor = (0, 0, 0)
        # 干扰线颜色。默认为红色
        self.linecolor = (0, 0, 0)
        # 是否要加入干扰线
        self.draw_line = drawline
        # 加入干扰线条数的上下限
        self.line_number = (1, 5)
        self.savepath = savepath
        self.rotate = rotate
        self.rangle = rangle
        pass

    def gen_one_char(self):
        """ 随机获得一个字符 """
        tmpletters = random.sample(self.source, 1)
        # print tmpletters[0]
        return tmpletters[0]
        pass

    def getcharlocation(self, lastchar_index, font_width, font_height):
        """ 获得待写字符的位置 """
        width, height = 35, 44
        c = 10 + 20 * lastchar_index        # 列号索引
        rowindex = height - 15 - font_height    # 底部需要留足空间用来旋转
        r = random.randint(5, rowindex if rowindex > 5 else 8)   #
        # r = 0
        # self.getrotatedloc(c, r, font_height, font_width)
        return r, c    # r,c font_width, font_height 可以作为训练集的box
        pass

    def pixelrotate(self, old_c, old_r, center_c, center_r, angle):
        old_c = old_c
        old_r = 44 - old_r
        center_c = center_c
        center_r = 44 - center_r
        new_c = (old_c - center_c) * math.cos(math.pi / 180.0 * angle) - (old_r - center_r) * math.sin(
            math.pi / 180.0 * angle) + center_c
        new_r = (old_c - center_c) * math.sin(math.pi / 180.0 * angle) + (old_r - center_r) * math.cos(
            math.pi / 180.0 * angle) + center_r

        new_c = int(math.ceil(new_c))
        new_r = int(math.ceil(44-new_r))
        return new_c, new_r

    def getrotatedloc(self, leftop_c, leftop_r, height, width, angle, center_c=70, center_r=22):
        """ 计算将原来的矩形旋转一定角度后,新矩形的坐标 """
        # 矩形中心点
        # center_c = leftop_c + width / 2
        # center_r = leftop_r + height / 2
        center_c = center_c     # 由于这里的旋转是整个图片旋转,所以旋转中心是图片中心点
        center_r = center_r
        # 右上的点
        rightop_c = leftop_c + width
        rightop_r = leftop_r

        # 左下的点
        leftbottom_c = leftop_c
        leftbottom_r = leftop_r + height

        # 右下的点
        rightbottom_c = leftop_c + width
        rightbottom_r = leftop_r + height

        lt_c, lt_r = self.pixelrotate(leftop_c, leftop_r, center_c, center_r, angle)
        rt_c, rt_r = self.pixelrotate(rightop_c, rightop_r, center_c, center_r, angle)
        lb_c, lb_r = self.pixelrotate(leftbottom_c, leftbottom_r, center_c, center_r, angle)
        rb_c, rb_r = self.pixelrotate(rightbottom_c, rightbottom_r, center_c, center_r, angle)
        new_lc = min([lt_c, rt_c, lb_c, rb_c])
        new_rc = max([lt_c, rt_c, lb_c, rb_c])
        new_lr = min([lt_r, rt_r, lb_r, rb_r])
        new_rr = max([lt_r, rt_r, lb_r, rb_r])
        return new_lc, 0, new_rc, 44
        pass

    def checkboxesright(self, box=None, img=None):
        """opencv画出框,检查box的位置是否正确, 肉眼观察"""
        imgdir = "captcha_6-char_test_30w_noline/"
        jsonpath = "captcha_6-char_test_30w_noline_label.json"
        imglist = os.listdir(imgdir)
        with open(jsonpath, "r")as rd:
            label_box_dic = json.loads(rd.read())
        for img in imglist:
            # print img
            imgpath = os.path.join(imgdir, img)
            imgobj = cv2.imread(imgpath)
            img_key = img.replace(".jpg", "")
            #if img in label_box_dic.keys():
            #	print "hit"
            boxlist =  label_box_dic[img_key]['boxlist']
            for i in range(6):
                cv2.rectangle(imgobj, tuple(boxlist[i][:2]), tuple(boxlist[i][2:]), (0,0,255), 1)
            imgsavepath = imgpath.replace(".jpg", "_box.jpg")
            cv2.imwrite(imgsavepath, imgobj)



    def getrandomfont(self, isdigit=False):
        """ 从字体池中随机抓一个字体 """
        self.fontpool = self.getfontyptlists()
        charfont = random.sample(self.fontpool, 1)[0]  # 从字体库取一个字体路径
        # print charfont
        font = ImageFont.truetype(charfont, 26)  # 验证码的字体
        return font
        pass

    def freshbox(self, imgobj, box):
        """
            提取图片中的文字的box
            原始box: [c, 4, c+font_width+6, 44]
        """
        row, col, channel = imgobj.shape
        row = row if row < 44 else 44
        col = col if col < 140 else 140
        cmin = 0
        rmin = 4
        cmax = col
        rmax = 44
        for c in range(col):
            if np.where(imgobj[:, c] != [255, 255, 255, 255]):
                cmin = c
                break

        for c in range(col)[::-1]:
            if np.where(imgobj[:, c] != [255, 255, 255, 255]):
                cmax = c
                break

        for r in range(row):
            if np.where(imgobj[r, :] != [255, 255, 255, 255]):
                rmin = r
                break

        for r in range(row)[::-1]:
            if np.where(imgobj[r, :] != [255, 255, 255, 255]):
                rmax = r
                break
        box = [box[0]+cmin, rmin, box[2]-(col-cmax), rmax]
        return box

        # print  np.where(timg[-20, -10] != [255, 255, 255, 255])
        # if np.where(timg[-20, -10] != [255, 255, 255, 255]) != np.array([]):
        #     print True
        # else:
        #     print False
        pass

    def writeonechar(self, tchar, tfont, tloc):
        """ 写入一个字符, 并且旋转一定角度, 同时修改幕布为白色 """

        pass


    def getrandomsin(self):
        """ 随机生成一个sin函数: y = Asin(wx + fie) + k"""
        h, w = (44, 140)
        circles = random.randint(1, 6)  # 图片中正弦曲线的周期波形个数
        T = math.ceil(float(w)/circles)
        w = 2 * math.pi / T
        A = math.ceil(circles * 3 / 10.0 * h)
        K = random.randint(math.floor(h/5.0), math.ceil(h/5.0 * 4 ))
        fie = random.randint(0, math.ceil(T/2.0))
        thickness = (7 - circles) / 2 # 实际上根据之前的统计，最粗的线的粗度在5左右，这里最粗的为6
        return lambda x: A * math.sin(w * x + fie) + K, thickness

        # if circles in [4, 5, 6]:
        #     thickness = random.randint(1, 2)
        # else:
        #     thickness = random.randint(3, 5)
        pass

    def gen_curseline(self, ans_image):
        """ 在验证码图片中生成正弦曲线 """
        # ans_image = Image.new('RGBA', (140, 44), self.bgcolor)  # 创建图片
        # ans_image = np.array(ans_image)
        func, thickness = self.getrandomsin()
        for i in range(140-1):
            startpoint = (i, int(round(func(i))))
            endpoint = (i+1, int(round(func(i+1))))
            cv2.line(ans_image, startpoint, endpoint, (0, 0, 0), thickness=thickness)
        cv2.line(ans_image, (0, int(func(0))),(139, int(func(39))), (0,0,0), thickness=1)
        # cv2.imshow("正弦曲线", ans_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pass

    def mycharotate(self, r, c, font_weight, font_height, angle):
        """
            字符旋转:不旋转整体图片,只旋转字符
            根据angle的正负判断旋转的方向,从而决定清除重写的顺序:正-从右边开始清除重写,负-从左侧开始清除重写
        """

        pass


    def rmeightNeibornoisepoint(self, imgobj, ridus):
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

            if black_count >= ridus:
                imgobj[r,c] = 0

            pass
        h, w = imgobj.shape
        for r in range(h):
            for c in range(w):
                if imgobj[r,c] >= 200:
                    pixel_8_neibor(r,c,imgobj)


    def gocaptchagenning(self, savefname):
        """
            代码里面涉及到numpy和PIL image的互换，这样方面image和opencv的组合使用，opencv中的逻辑操作还是很有用的，bitwise_and....
            还尝试很多图片合并粘贴的操作，但是opencv的位运算效果更好

            说明一下：h*w =  44*140的宽度分配规则是：左右两侧分别留出10p, 剩下120p均匀分给6个字符，各20p，这个规则用于计算每个字符的起始col值
            然而在实际生成填写单个字符的小图片时，宽度给的是font_width+6, 也就是说，实际的写单个字符的小图片宽度范围是[c, c+font_width+6]
            并且在写字符的时候，左右各留出3p，为了给字符旋转腾出空间，
            因此，由于font_width的变化，可能出现font_width+6 > 20的情况，也就是字符粘黏
        """
        width, height = self.size  # 宽和高
        ans_image = Image.new('RGBA', (width, height), self.bgcolor)  # 创建图片
        ans_image = np.array(ans_image)
        # ans_image =
        result_chars = []
        boxlist= []
        for i in range(self.number):
            # 任选一个字体
            font = self.getrandomfont()

            # 生成单个字符
            text = self.gen_one_char()
            # 计算字符放置坐标
            font_width, font_height = font.getsize(text)
            # print "font_width:%s, font_height:%s" % (font_width, font_height)
            r, c = self.getcharlocation(i, font_width, font_height)
            # 创建图片
            image = Image.new('RGBA', (font_width+6, height), self.bgcolor)
            # 创建画笔
            draw = ImageDraw.Draw(image)
            # 填充字符串
            draw.text((3, r), text, font=font, fill=self.fontcolor)

            # print "origin_c:%s, origin_r:%s" % (c, r)
            # print "font_width:%s, font_height:%s" %(font_width, font_height)


            # 创建扭曲
            # image = image.transform((width + 20, height + 10), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0), Image.BILINEAR)
            # fff = Image.new('RGBA', image.size, (255,) * 4)
            # image = Image.composite(image, fff, image)
            # print "new Image shape:" , image.size
            box = [c, 4, c+font_width+6, 44]    # 因为r是（5+）起步的，所以这里给4是安全的
            if self.rotate:
                # 字符旋转
                # image.save("tmp1.png")
                angle = random.randint(-self.rangle, self.rangle)  # -10, 15
                image = image.rotate(angle, expand=0)     # random.randint(-10, 5)
                # 将旋转后漏出的幕布用白色填充
                fff = Image.new('RGBA', image.size, (255,) * 4)
                image = Image.composite(image, fff, image)

                # 滤镜，边界加强
                image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                # 更新旋转后的box
                # box = self.getrotatedloc(c, r, font_height, font_width, angle=angle)
                timg = np.array(image)
                box = self.freshbox(timg, box)
                # print "旋转后的image.shape:" , image.size

            # 将图片粘到ans_image上
            image = np.array(image)
            # print image.shape[1]
            # print image.shape
            h, w, channel = image.shape
            if c+w > 139:
                dis = (c+w) - 139
                image = image[:, :w-dis]
                step = 139-c
            else:
                step = w
            # step = w if c+w < 140 else 139-c

            tarea = ans_image[0:44, c:c+step]		# 部分字体会由于图片旋转，导致image的width变化，因为新建小图片的时候没有验证c+font_width+6是否大于140
            try:
                ans_image[0:44, c:c+step] = cv2.bitwise_and(tarea, image)
            except Exception as e:
                print "type(tarea):%s, type(image):%s" % (type(tarea), type(image))
                print "tarea.shape:%s, image.shape:%s" % (tarea.shape, image.shape)
            # 将旋转后的字母区域剪切粘贴到ans_image
            # print angle, box
            # region = image.crop(box)
            # print region
            # ans_image.paste(region, box)
            # ans_image.paste(image, box)

            boxlist.append(box)
            # image.save('tmp.png')
            # 滤镜，边界加强
            # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            result_chars.append(text)

        # 降噪: 主要是边缘平滑
        # kernel = np.ones((3,3), np.uint8)
        # ans_image = cv2.morphologyEx(ans_image, cv2.MORPH_CLOSE, kernel)

        im_gray = cv2.cvtColor(ans_image, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
        retval, ans_image = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY) # 二值化
        # self.rmeightNeibornoisepoint(ans_image, 6)	# 8领域降噪
        #ans_image = cv2.medianBlur(ans_image,3)

        #ans_image = cv2.pyrUp(ans_image)
        #for i in range(15):
        #    ans_image = cv2.medianBlur(ans_image, 3)
        #ans_image = cv2.pyrDown(ans_image)
        #retval, ans_image = cv2.threshold(ans_image, 200, 255, cv2.THRESH_BINARY) # 二值化

        ## 加干扰线
        if self.draw_line:
            self.gen_curseline(ans_image)       # opencv 画曲线
            # self.gene_line(draw, width, height) # PIL Image 画曲线

        cv2.imwrite(savefname, ans_image)
        # ans_image = Image.fromarray(ans_image.astype('uint8')).convert("RGB")
        # ans_image = ans_image.convert("RGB")
        #ans_image.save(savefname)  # 保存验证码图片
        print "finished gen %s......" % savefname
        return "".join(result_chars).lower(), boxlist

    def run(self):
        label_dic = {}
        # 获取当前目录和标记文件的绝对路径
        label_filename = os.path.basename(self.savepath)	# 保存图片的文件夹名称
        localdirpath = os.path.abspath(".")			# 当前文件夹的绝对路径
        labelfilepath = os.path.join(localdirpath, "%s_label.json" % label_filename)  # 标记文件的名称: img文件夹名+label.json 

        # 
        if os.path.exists(labelfilepath):
            with open(labelfilepath)as rd:
                label_dic = json.loads(rd.read())
            startindex = max([int(k) for k in label_dic.keys()])
        else:
            startindex = 0
        for c in range(startindex, startindex+self.totalnum):
            label, boxlist = self.gocaptchagenning(os.path.join(self.savepath, "%s.jpg" % c))
            label_dic[c] = {"label": label, "boxlist": boxlist}

        with open(labelfilepath, "w")as wr:
            wr.write(json.dumps(label_dic))
        pass


    def getfontyptlists(self, isdigit=False):
        """ 获取ubuntu16.04中所有的系统字体"""
        anslist = []
        filterlist = []
        with open('font.txt','r')as rd:
            lines= rd.readlines()
        filterlist = [line.strip() for line in lines]
        d_filter = """z003034l_pfb,n021023l_pfb,Purisa_ttf,Purisa-Bold_ttf""".split(',')
        if isdigit:
            filterlist.extend(d_filter)
        for root, dirs, files in os.walk("/usr/share/fonts"):
            # print "root:%s" % root
            # print "dirs:%s" % dirs
            # print "files%s" % files
            anslist.extend([os.path.join(root, f) for f in files if f.replace('.', '_') in filterlist]) #  and os.path.splitext(f)[1] in ['.ttf', '.ttc']
        return anslist
        pass

    def printusefontpool(self):
        """ 生成指定字体的图片 """
        fontpool = self.getfontyptlists()
        for cindex, font in enumerate(fontpool):
            text = '2'
            imgfilename = os.path.basename(font)
            imgfilename = imgfilename.replace(".", "_")
            # if imgfilename not in """Purisa-BoldOblique_ttf,Purisa-Bold_ttf,Purisa-Oblique_ttf,Purisa_ttf""".split(","):
            #     continue
            font = ImageFont.truetype(font, 25)
            font_width, font_height = font.getsize(text)
            # print "font_width:%s, font_height:%s" % (font_width, font_height)
            r, c = self.getcharlocation(0, font_width, font_height)
            # 创建图片
            image = Image.new('RGBA', (font_width + 20, 44), self.bgcolor)
            # 创建画笔
            draw = ImageDraw.Draw(image)
            # 填充字符串
            draw.text((3, r), text, font=font, fill=self.fontcolor)
            image = image.convert("RGB")

            # print imgfilename
            image.save(os.path.join("captchaV2", "%s.jpg"% imgfilename))
        pass

if __name__ == '__main__':
    # 测试gencaptcha_cnn_98
    # tobj = gencaptcha_cnn_98()
    # count = 0
    # while(count<2):
    #     text, image = tobj.gen_captcha_text_and_image()
    #     # print 'begin ', time.ctime(), type(image)
    #     f = plt.figure()
    #     ax = f.add_subplot(111)
    #     ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
    #     plt.imshow(image)
    #     plt.show()
    #     count += 1

    # 测试gencapthca_test
    # tobj = gencapthca_test()
    # tobj.gene_code()

    # 测试gencaptcha_final
    tobj = gencaptcha_final(6, 300000, savepath="/home/jingdata/Document/LAB_CODE/captcha/Captcha_Recg/captcha_6-char_test_30w_noline", rotate=True, drawline=True)
    # tobj = gencaptcha_final(mode=6, totalnum=120000, savepath="/home/jingdata/Document/LAB_CODE/captcha/Captcha_Recg/captcha_1-char_12w", rotate=True)
    tobj.run()
    # tobj.gen_curseline("")
    # 测试字体打印
    # tobj.printusefontpool()
    # tobj.checkboxesright()


    # 初始化停用词文件
    # tlist = list(string.letters)
    # for index in range(0, 10):
    #     tlist.append(str(index))
    # with open("/home/jingdata/桌面/stop_words.txt", "w")as wr:
    #     for i in tlist:
    #         wr.write(i)
    #         wr.write('\n')
    # 读取停用词
    #stop_words = []
    #with open("/home/jingdata/桌面/stop_words.txt", "r")as rd:
    #    lines = rd.readlines()
    #for line in lines:
    #    if line.strip():
    #       stop_words.append(line.strip())
    #print stop_words
    #print len(stop_words)
