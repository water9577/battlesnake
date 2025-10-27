# 文件名：main.py（Replit部署核心）
import os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify

# 🔹 1. 加载训练好的PPO模型
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.actor(x), dim=1)

# 初始化模型（加载本地训练的权重）
model = PolicyNetwork()
model.load_state_dict(torch.load("ppo_snake_model.pth", map_location=torch.device))
model.eval()  # 推理模式

# 🔹 2. 状态预处理（复刻PDF的环境特征提取）
def preprocess_state(game_state):
    my_snake = game_state['you']
    head = my_snake['head']
    food = game_state['board']['food'][0] if game_state['board']['food'] else head
    enemies = game_state['board']['snakes']
    enemy_head = enemies[0]['head'] if enemies else head  # 无敌方则用自身头部填充

    # 8维状态向量（对应训练端）
    return np.array([
        head['x'] - food['x'],
        head['y'] - food['y'],
        abs(head['x'] - food['x']) + abs(head['y'] - food['y']),
        1 if head['x'] < food['x'] else 0,
        enemy_head['x'],
        enemy_head['y'],
        len(my_snake['body']),
        len(enemies[0]['body']) if enemies else len(my_snake['body'])
    ], dtype=np.float32)

# 🔹 3. 启发式避障（PDF的“工程化容错”思路，防止模型失误）
def get_safe_moves(game_state):
    my_snake = game_state['you']
    head = my_snake['head']
    body = my_snake['body']
    board = game_state['board']
    possible_moves = ['up', 'down', 'left', 'right']
    safe = []

    for move in possible_moves:
        new_head = head.copy()
        if move == 'up': new_head['y'] += 1
        elif move == 'down': new_head['y'] -= 1
        elif move == 'left': new_head['x'] -= 1
        elif move == 'right': new_head['x'] += 1

        # 检查撞墙/撞自身
        if (new_head['x'] < 0 or new_head['x'] >= board['width'] or
            new_head['y'] < 0 or new_head['y'] >= board['height'] or
            new_head in body[:-1]):
            continue
        safe.append(move)
    return safe

# 🔹 4. 核心决策（融合PPO模型+启发式）
def move(game_state):
    # 第一步：获取安全方向（基础容错）
    safe_moves = get_safe_moves(game_state)
    if not safe_moves:
        return {'move': 'up'}  # 保底方向

    # 第二步：PPO模型预测
    state = preprocess_state(game_state)
    state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
    action_probs = model(state_tensor)
    action_idx = torch.argmax(action_probs).item()
    model_move = ['up', 'down', 'left', 'right'][action_idx]

    # 第三步：优先选择“模型推荐+安全”的方向（PDF的策略融合）
    return {'move': model_move if model_move in safe_moves else safe_moves[0]}

# 🔹 5. Flask服务（对接Battlesnake API）
app = Flask(__name__)
@app.route('/', methods=['GET'])
def info():
    return jsonify({
        "apiversion": "1",
        "author": "你的团队名",
        "color": "#4285F4",
        "head": "brain",
        "tail": "bolt"
    })

@app.route('/start', methods=['POST'])
def start():
    return jsonify({"ok": True})

@app.route('/move', methods=['POST'])
def handle_move():
    return jsonify(move(request.get_json()))

@app.route('/end', methods=['POST'])
def end():
    return jsonify({"ok": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
