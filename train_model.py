#!/usr/bin/env python3
"""
Script training model DQN
Chạy độc lập, không cần web interface
"""

import os
import json
import time
from datetime import datetime
from playlist_generator import PlaylistGenerator
from models import DQNModel, PlaylistEnvironment
from config import Config

def add_training_log(message, logs_list):
    """Thêm log training với timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    logs_list.append(log_entry)
    print(log_entry, flush=True)

def main():
    print("🧠 KIỂM TRA VÀ TRAINING MODEL DQN")
    print("=" * 50)
    
    # Kiểm tra dữ liệu
    if not os.path.exists(Config.PLAYLIST_DATA_FILE):
        print("❌ Không tìm thấy dữ liệu!")
        print(f"   Vui lòng chạy collect_data.py trước")
        return
    
    # Kiểm tra model đã có chưa
    model_path = 'models/dqn_model.h5'
    if os.path.exists(model_path):
        print(f"✅ Đã tìm thấy model hiện có: {model_path}")
        
        # Kiểm tra thông tin model
        try:
            generator = PlaylistGenerator()
            if generator.load_data():
                print(f"📊 Dữ liệu: {len(generator.songs_data)} bài hát")
                
                choice = input("\n❓ Bạn có muốn training lại model không? (y/N): ").lower().strip()
                if choice != 'y':
                    print("✅ Giữ nguyên model hiện có!")
                    print("🎉 Hệ thống đã sẵn sàng để sử dụng!")
                    return
                else:
                    print("🔄 Bắt đầu training lại model...")
            else:
                print("⚠️ Không thể load dữ liệu, bắt đầu training mới...")
        except Exception as e:
            print(f"⚠️ Lỗi khi kiểm tra model: {e}")
            print("🔄 Bắt đầu training mới...")
    else:
        print("📊 Chưa có model, bắt đầu training...")
    
    # Khởi tạo generator
    print("\n📊 Đang load dữ liệu...")
    generator = PlaylistGenerator()
    if not generator.load_data():
        print("❌ Không thể load dữ liệu!")
        return
    
    print(f"✅ Đã load {len(generator.songs_data)} bài hát")
    
    # Hỏi số episodes
    try:
        episodes = int(input("Nhập số episodes training (mặc định 20): ") or "20")
    except ValueError:
        episodes = 20
    
    print(f"🎯 Sẽ training {episodes} episodes")
    print()
    
    # Bắt đầu training
    training_logs = []
    
    try:
        # Tạo embeddings nếu chưa có
        if not generator.embeddings:
            print("🔄 Đang tạo embeddings...")
            add_training_log("Tạo embeddings cho bài hát...", training_logs)
            generator.create_embeddings()
            add_training_log("Embeddings hoàn thành!", training_logs)
            print()
        
        # Tạo environment
        print("🔄 Đang tạo environment...")
        add_training_log("Khởi tạo environment và model...", training_logs)
        generator.environment = PlaylistEnvironment(generator.songs_data, generator.embeddings)
        print(f"✅ Environment có {len(generator.environment.available_songs)} bài hát available")
        
        # Tạo DQN model
        state_size = Config.EMBEDDING_DIM
        action_size = len(generator.songs_data)
        generator.dqn_model = DQNModel(state_size, action_size)
        print(f"✅ DQN model: state_size={state_size}, action_size={action_size}")
        print()
        
        # Bắt đầu training
        add_training_log(f"Bắt đầu training {episodes} episodes...", training_logs)
        print("=" * 60)
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Log progress
            if episode == 0 or episode % 10 == 0:
                progress = (episode / episodes) * 100
                add_training_log(f"Episode {episode}/{episodes} ({progress:.1f}%)", training_logs)
            
            # Train một episode
            try:
                state = generator.environment.reset()
                total_reward = 0
                steps = 0
                
                while True:
                    try:
                        action = generator.dqn_model.act(state, generator.environment.available_songs)
                        next_state, reward, done, _ = generator.environment.step(action)
                        generator.dqn_model.remember(state, action, reward, next_state, done)
                        
                        state = next_state
                        total_reward += reward
                        steps += 1
                        
                        if len(generator.dqn_model.memory) > 32:
                            generator.dqn_model.replay(32)
                        
                        if done:
                            break
                            
                    except Exception as step_error:
                        add_training_log(f"Lỗi trong step {steps} của episode {episode}: {str(step_error)}", training_logs)
                        break
                
                # Log kết quả episode
                add_training_log(f"Episode {episode}: Hoàn thành với {steps} steps, reward = {total_reward:.2f}", training_logs)
                
                # Log chi tiết mỗi 10 episodes (trừ episode 0)
                if episode % 10 == 0 and episode > 0:
                    add_training_log(f"Episode {episode}: Total Reward = {total_reward:.2f}, Steps = {steps}, Epsilon = {generator.dqn_model.epsilon:.3f}", training_logs)
                    add_training_log(f"Memory size: {len(generator.dqn_model.memory)}, Available songs: {len(generator.environment.available_songs)}", training_logs)
                    print("-" * 40)
                    
            except Exception as e:
                add_training_log(f"Lỗi trong episode {episode}: {str(e)}", training_logs)
                continue
        
        # Hoàn thành
        end_time = time.time()
        training_time = end_time - start_time
        
        add_training_log("Training hoàn thành!", training_logs)
        
        # Lưu model
        add_training_log("Lưu model...", training_logs)
        os.makedirs('models', exist_ok=True)
        generator.dqn_model.save('models/dqn_model.h5')
        add_training_log("Model đã được lưu thành công!", training_logs)
        
        # Thống kê cuối cùng
        add_training_log("Thống kê cuối cùng:", training_logs)
        add_training_log(f"   - Tổng episodes: {episodes}", training_logs)
        add_training_log(f"   - Thời gian training: {training_time:.2f} giây", training_logs)
        add_training_log(f"   - Epsilon cuối: {generator.dqn_model.epsilon:.3f}", training_logs)
        add_training_log(f"   - Memory size: {len(generator.dqn_model.memory)}", training_logs)
        
        print()
        print("✅ TRAINING HOÀN THÀNH THÀNH CÔNG!")
        print(f"📁 Model được lưu tại: models/dqn_model.h5")
        print(f"⏱️  Thời gian training: {training_time:.2f} giây")
        
    except Exception as e:
        print()
        print(f"❌ LỖI TRAINING: {str(e)}")
        add_training_log(f"Lỗi training: {e}", training_logs)

if __name__ == "__main__":
    main() 