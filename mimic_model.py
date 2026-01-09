import torch
import torch.nn as nn
from attention import SpatioChannelAttention, ChannelAttention, SpatialAttention

class FeatureMimickingFramework(nn.Module):
    """
    Implements the architecture in Figure 1[cite: 106].
    """
    def __init__(self, teacher_model, student_model, attention_type='SCA', feat_channels=512):
        super(FeatureMimickingFramework, self).__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        # Freeze Teacher parameters 
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Select Attention Module
        if attention_type == 'SCA':
            self.attention = SpatioChannelAttention(feat_channels)
        elif attention_type == 'CA':
            self.attention = ChannelAttention(feat_channels)
        elif attention_type == 'SA':
            self.attention = SpatialAttention()
        else:
            raise ValueError("Invalid attention type")

    def forward(self, x):
        # 1. Get Teacher Features (Fea_T) [cite: 94]
        with torch.no_grad():
            teacher_output, fea_t = self.teacher(x)
        
        # 2. Get Student Features (Fea_S) before refinement [cite: 103]
        # Note: Student model needs to return both final output and intermediate feature
        student_output_raw, fea_s = self.student(x, refine=False)
        
        # 3. Calculate Feature Inconsistency: Fea_I = |Fea_T - Fea_S| 
        fea_i = torch.abs(fea_t - fea_s)
        
        # 4. Generate Attention Map [cite: 125, 128]
        att_map = self.attention(fea_i)
        
        # 5. Refine Feature: Fea_R = Fea_S * Map (Element-wise multiplication)
        # Note: The paper says "refines the student feature". 
        # Usually typical attention involves (1 + map) * feature or map * feature.
        # Based on Fig 1 flow: Map -> (x) -> Fea_R. We assume element-wise mul.
        fea_r = fea_s * att_map 
        
        # 6. Student continues follow-up tasks with Refined Feature [cite: 109]
        # You need a method in your student model to decode from the hidden feature
        final_output = self.student.decode_from_feature(fea_r)
        
        return {
            'output': final_output,
            'fea_t': fea_t,
            'fea_s': fea_s,
            'fea_i': fea_i,
            'fea_r': fea_r,
            'att_map': att_map
        }
