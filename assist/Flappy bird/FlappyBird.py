import pygame
import random
# 初始化pygame
pygame.init()

# 游戏参数
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
FPS = 30

# 创建游戏窗口
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

score = 0
# 定义小鸟类
class Bird():
    def __init__(self):
        super().__init__()
        # 加载小鸟图像
        self.image = pygame.image.load('resizedbird.png')
        self.rect = self.image.get_rect()
        self.rect.center = (50, SCREEN_HEIGHT // 2)
        self.velocity = 0

    def jump(self):
        self.velocity = -8

    def update(self):
        self.velocity += 1
        self.rect.y += self.velocity


# 定义柱子类
class PipePair(pygame.sprite.Sprite):
    def __init__(self, x):
        super().__init__()
        self.image_top = pygame.image.load('resizedpipe_up.png')
        self.image_bottom = pygame.image.load('resizedpipe_bottom.png')
        self.rect_top = self.image_top.get_rect()
        self.rect_bottom = self.image_bottom.get_rect()
        self.gap = random.randint(120,180)  # 通道间隙的大小
        self.x = x
        self.passed = False

        # 设置柱子位置
        self.topMove = random.randint(0,200)
        self.bottomMove = random.randint(0,50)


    def update(self):
        self.x -= 5
        self.rect_top.topleft = (self.x, 0 - self.topMove)
        self.rect_bottom.topleft = (self.x, 300 - self.topMove + self.gap)

#定义游戏环境类，基本上游戏的操作都在这里
class Mygame():
    def __init__(self):
        self.background_image = pygame.image.load('background.png')
        self.done = None
    def reset(self):
        self.bird = Bird()
        self.pipes = pygame.sprite.Group()
        self.old_time = pygame.time.get_ticks()
        self.done = False
        global score
        score = 0
    def reward(self):
        for pipe in self.pipes:
            if pipe.x + pipe.rect_top.width < 0:
                pipe.kill()
            if pipe.x + pipe.rect_top.width < self.bird.rect.left and not pipe.passed:
                pipe.passed = True
                global score
                score += 1
                return 20   #如果成功通过柱子reward是+20
        for pipe_pair in self.pipes:
            screen.blit(pipe_pair.image_top, (pipe_pair.x, 0 - pipe_pair.topMove))
            screen.blit(pipe_pair.image_bottom, (pipe_pair.x, 300 - pipe_pair.topMove + pipe_pair.gap))
            if self.bird.rect.colliderect(pipe_pair.rect_top) or self.bird.rect.colliderect(pipe_pair.rect_bottom):
                #这一回合结束
                self.done = True
                return -100
        if self.bird.rect.top < 0 or self.bird.rect.bottom > SCREEN_HEIGHT:
            self.done = True
            return -300
        return 0


    def step(self):
        current_time = pygame.time.get_ticks()
        # 添加柱子对
        if current_time - self.old_time > 1720:  # 控制屏幕上同时出现的柱子对数量
            new_pipe = PipePair(SCREEN_WIDTH)
            self.pipes.add(new_pipe)
            self.old_time = current_time
        self.bird.update()
        self.pipes.update()
        # 绘制背景、小鸟和柱子
        screen.blit(self.background_image, (0, 0))
        screen.blit(self.bird.image, self.bird.rect)
        font = pygame.font.Font(None, 36)  # 设置字体和字号
        text = font.render(f'Score: {score}', True, (0, 0, 0))  # 渲染得分文本
        screen.blit(text, (10, 10))  # 显示得分文本的位置

        # 游戏逻辑更新
        #pygame.display.flip()
        #clock.tick(FPS)