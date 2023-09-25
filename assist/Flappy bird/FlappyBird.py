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

# 定义小鸟类
class Bird(pygame.sprite.Sprite):
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
        self.gap = random.randint(100,200)  # 通道间隙的大小
        self.x = x
        self.passed = False

        # 设置柱子位置
        self.topMove = random.randint(0,200)
        self.bottomMove = random.randint(0,50)


    def update(self):
        self.x -= 5
        self.rect_top.topleft = (self.x, 0 - self.topMove)
        self.rect_bottom.topleft = (self.x, 300 - self.topMove + self.gap)
        if self.x + self.rect_top.width < 0:
            self.kill()
        if self.x + self.rect_top.width < bird.rect.left and not self.passed:
            self.passed = True
            global score
            score += 1

# 初始化小鸟和柱子组
bird = Bird()
pipes = pygame.sprite.Group()
old_time = pygame.time.get_ticks()
background_image = pygame.image.load('background.png')
reward = 0
score = 0
# 游戏循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird.jump()
    current_time = pygame.time.get_ticks()
    # 添加柱子对
    if current_time - old_time > 1500:  # 控制屏幕上同时出现的柱子对数量
        new_pipe = PipePair(SCREEN_WIDTH)
        pipes.add(new_pipe)
        old_time = current_time

    bird.update()
    pipes.update()

    # 绘制背景、小鸟和柱子
    screen.blit(background_image, (0, 0))
    screen.blit(bird.image, bird.rect)

    for pipe_pair in pipes:
        screen.blit(pipe_pair.image_top, (pipe_pair.x, 0-pipe_pair.topMove))
        screen.blit(pipe_pair.image_bottom, (pipe_pair.x, 300 - pipe_pair.topMove + pipe_pair.gap))
        if bird.rect.colliderect(pipe_pair.rect_top) or bird.rect.colliderect(pipe_pair.rect_bottom):
            running = False
            print("over, Score = {}".format(score))

    font = pygame.font.Font(None, 36)  # 设置字体和字号
    text = font.render(f'Score: {score}', True, (0, 0, 0))  # 渲染得分文本
    screen.blit(text, (10, 10))  # 显示得分文本的位置

    # 游戏逻辑更新  


    pygame.display.flip()
    clock.tick(FPS)
pygame.quit()