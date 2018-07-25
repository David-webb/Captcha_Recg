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
import copy

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
    def __init__(self, mode, totalnum, savepath="./captcha_set1", rotate=False, rangle=45):
        self.source = list(string.letters)
        for index in range(0, 10):
            self.source.append(str(index))

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
        self.draw_line = True
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
        rowindex = height - 15- font_height    # 底部需要留足空间用来旋转
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

    def getrandomfont(self):
        """ 从字体池中随机抓一个字体 """
        charfont = random.sample(self.fontpool, 1)[0]  # 从字体库取一个字体路径
        font = ImageFont.truetype(charfont, 25)  # 验证码的字体
        return font
        pass

    def writeonechar(self, tchar, tfont, tloc):
        """ 写入一个字符, 并且旋转一定角度, 同时修改幕布为白色 """

        pass

    def gen_curseline(self):
        """ 在验证码图片中生成正弦曲线 """
        pass

    def mycharotate(self, r, c, font_weight, font_height, angle):
        """
            字符旋转:不旋转整体图片,只旋转字符
            根据angle的正负判断旋转的方向,从而决定清除重写的顺序:正-从右边开始清除重写,负-从左侧开始清除重写
        """

        pass



        pass
    def gocaptchagenning(self, savefname):
        """ """
        width, height = self.size  # 宽和高
        ans_image = Image.new('RGBA', (width, height), self.bgcolor)  # 创建图片
        result_chars = []
        boxlist= []
        for i in range(self.number):
            image = Image.new('RGBA', (width, height), self.bgcolor)  # 创建图片
            # 任选一个字体
            font = self.getrandomfont()
            # 创建画笔
            draw = ImageDraw.Draw(image)
            # 生成单个字符
            text = self.gen_one_char()
            # 计算字符放置坐标
            font_width, font_height = font.getsize(text)
            # print "font_width:%s, font_height:%s" % (font_width, font_height)
            r, c = self.getcharlocation(i, font_width, font_height)
            # 填充字符串
            draw.text((c, r), text, font=font, fill=self.fontcolor)
            # print "origin_c:%s, origin_r:%s" % (c, r)
            # print "font_width:%s, font_height:%s" %(font_width, font_height)
            # 加干扰线
            # if self.draw_line:
            #     self.gene_line(draw, width, height)

            # 创建扭曲
            # image = image.transform((width + 20, height + 10), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0), Image.BILINEAR)
            # fff = Image.new('RGBA', image.size, (255,) * 4)
            # image = Image.composite(image, fff, image)

            box = [c, 0, c+font_height, 44]
            if self.rotate:
                # 字符旋转
                # image.save("tmp1.png")
                angle = random.randint(-20, 20)  # -10, 15
                image = image.rotate(angle, expand=0)     # random.randint(-10, 5)
                # 将旋转后漏出的幕布用白色填充
                fff = Image.new('RGBA', image.size, (255,) * 4)
                image = Image.composite(image, fff, image)
                # 将旋转后的字母区域剪切粘贴到ans_image
                box = self.getrotatedloc(c, r, font_height, font_width, angle=angle)
                # print angle, box
                region = image.crop(box)
                ans_image.paste(region, box)
            boxlist.append(box)
            # image.save('tmp.png')
            # 滤镜，边界加强
            # image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)

            result_chars.append(text)
	    ans_image = ans_image.convert("RGB")
        ans_image.save(savefname)  # 保存验证码图片
        print "finished gen %s......" % savefname
        return "".join(result_chars).lower(), boxlist

    def run(self):
        label_dic = {}
        label_filename = os.path.basename(self.savepath)
        localdirpath = os.path.abspath("..")
        labelfilepath = os.path.join(localdirpath, "%s_label.json" % label_filename)
        if os.path.exists(labelfilepath):
            with open(labelfilepath)as rd:
                label_dic = json.loads(rd.read())
            startindex = int(max(label_dic.keys()))
        else:
            startindex = 0

        for c in range(startindex, startindex+self.totalnum):
            label, boxlist = self.gocaptchagenning(os.path.join(self.savepath, "%s.jpg" % c))
            label_dic[c] = {"label":label, "boxlist": boxlist}

        with open(labelfilepath, "w")as wr:
            wr.write(json.dumps(label_dic))
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
    tobj = gencaptcha_final(6, 5, savepath="/home/jingdata/Document/LAB_CODE/captcha/Captcha_Recg/captcha_6-char_test", rotate=True)
    # tobj = gencaptcha_final(mode=6, totalnum=120000, savepath="/home/jingdata/Document/LAB_CODE/captcha/Captcha_Recg/captcha_1-char_12w", rotate=True)
    tobj.run()

    # tobj.gocaptchagenning()
