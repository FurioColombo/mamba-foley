---
layout: default
title:  "MambaFoley: Foley Sound Generation using Selective State-Space Models"
---

Accompanying website to the paper _MambaFoley: Foley Sound Generation using Selective State-Space Models, Marco Furio Colombo, Francesca Ronchini, Luca Comanducci, Fabio Antonacci, submitted at ICASSP 2024.

# Abstract
Recent advancements in deep learning have led to widespread use of techniques for audio content generation, notably employing Denoising Diffusion Probabilistic Models (DDPM) across various tasks.  Among these, Foley Sound Synthesis is of particular interest for its role in applications for the creation of multimedia content. 
Given the temporal-dependent nature of sound, it is crucial to design generative models that can effectively handle the sequential modeling of audio samples. Selective State Space Models (SSMs) have recently been proposed as a valid alternative to previously proposed techniques, demonstrating competitive performance with lower computational complexity.
In this paper, we introduce MambaFoley, a diffusion-based model that, to the best of our knowledge, is the first to leverage the recently proposed SSM known as Mamba for the Foley sound generation task.
To evaluate the effectiveness of the proposed method, we compare it with a state-of-the-art Foley sound generative model using both objective and subjective analyses. 

# Audio Examples
In this page, we present audio samples generated using our model `MambaFoley` with samples generated  using `T-Foley` and `AttentionFoley`.
In order to present a relevant and significant comparison between generative models, we offer a comparison between seven different categories generated with the same temporal conditioning.

## 1) DogBark conditioned samples
<div style="display: flex; align-items: center;">
  <span>Conditioning audio: </span>
  <audio src="audio/ConditionedAudio/DogBarkConditioned/GroundTruths/cond_dogbark.wav" controls preload style="width: 250px; margin-left: 25px;"></audio>
</div>

### Dog Barking
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1" style="text-align: center;">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/mamba/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/lstm/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/attn/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Foot Steps
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1" style="text-align: center;">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/mamba/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/lstm/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/attn/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Gunshots
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/mamba/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/lstm/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/attn/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Keyboard
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/mamba/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/lstm/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/attn/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### MovingMotorVehicle
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/mamba/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/lstm/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/attn/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Rain
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/mamba/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/lstm/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/attn/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Sneezes and Coughs
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/mamba/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/lstm/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/DogBarkConditioned/attn/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>



## 2) Rain conditioned samples
<div style="display: flex; align-items: center;">
  <span>Conditioning audio: </span>
  <audio src="audio/ConditionedAudio/RainConditioned/GroundTruths/cond_rain.wav" controls preload style="width: 250px; margin-left: 25px;"></audio>
</div>

### Dog Barking
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1" style="text-align: center;">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/mamba/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/lstm/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/attn/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Foot Steps
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1" style="text-align: center;">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/mamba/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/lstm/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/attn/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Gunshots
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/mamba/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/lstm/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/attn/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Keyboard
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/mamba/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/lstm/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/attn/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### MovingMotorVehicle
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/mamba/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/lstm/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/attn/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Rain
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/mamba/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/lstm/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/attn/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Sneezes and Coughs
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/mamba/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/lstm/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/RainConditioned/attn/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>



## 3) Voice conditioned samples
<div style="display: flex; align-items: center;">
  <span>Conditioning audio: </span>
  <audio src="audio/ConditionedAudio/CountConditioned/GroundTruths/cond_count.wav" controls preload style="width: 250px; margin-left: 25px;"></audio>
</div>

### Dog Barking
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1" style="text-align: center;">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/mamba/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/lstm/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/attn/DogBark/DogBark_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Foot Steps
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1" style="text-align: center;">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/mamba/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/lstm/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/attn/Footstep/Footstep_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Gunshots
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/mamba/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/lstm/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/attn/GunShot/GunShot_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Keyboard
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/mamba/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/lstm/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/attn/Keyboard/Keyboard_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### MovingMotorVehicle
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/mamba/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/lstm/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/attn/MovingMotorVehicle/MovingMotorVehicle_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Rain
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/mamba/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/lstm/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/attn/Rain/Rain_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>

### Sneezes and Coughs
<div class="container" style="display: flex; justify-content: space-between; margin-bottom: 20px;">
   <div class="column-1">
     <h5>MambaFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/mamba/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-2" style="text-align: center;">
     <h5>T-Foley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/lstm/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
   <div class="column-3" style="text-align: center;">
     <h5>AttentionFoley</h5>
     <audio src="audio/ConditionedAudio/CountConditioned/attn/Sneeze_Cough/Sneeze_Cough_001.wav" controls preload style="width: 250px;"></audio>
   </div>
</div>
