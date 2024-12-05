from transformers import BertModel,BertConfig,AutoModel
from config_cross import *
import torch
from torch import nn
import torch.nn.functional as F
from lxrt import LXRTEncoder
from CLIP import *
# from VLMO import *

class RumorDetectionModel(nn.Module):  # 模型这块
    def __init__(self,                 
                 args,
                lang,
                loss_name='ce'):
        super().__init__() 
        self.loss_name = loss_name  # 损失函数的名字
        if lang =='zh':
            self.vision_encoder = vit(args.vision_backbone_zh)  # clip-vit
#             self.vision_encoder = vlmo_base_patch16(args.vision_backbone_zh)
            self.text_encoder = AutoModel.from_pretrained(args.bert_dir_zh)  # roberta
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_zh+"/config.json")  # 加载roberta的配置文件
        else:
            self.vision_encoder = vit(args.vision_backbone_en)  # clip-vit
            self.text_encoder = AutoModel.from_pretrained(args.bert_dir_en)  # deberta 
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_en+"/config.json")
        self.cross_encoder = LXRTEncoder(self.bertconfig)  # TODO 交叉注意力机制 交叉注意力做了几层
        self.fusion_num = 2  
        self.meanpooling = nn.AdaptiveAvgPool2d((1,self.text_encoder.config.hidden_size))
        # self.mid_linear = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,self.text_encoder.config.hidden_size//2),
        #                                 nn.Dropout(0.1))
        self.logits = nn.ModuleList([nn.Linear(self.text_encoder.config.hidden_size, 3) for i in range(self.fusion_num)])
        # self.logits = nn.Linear(self.text_encoder.config.hidden_size, len(CATEGORY_ID_LIST))
        vision_dim = 512
        self.projection = nn.Sequential(nn.Linear(vision_dim,self.text_encoder.config.hidden_size))
        self.encoder = nn.GRU(input_size=vision_dim, hidden_size=vision_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.encoder.flatten_parameters() # 这个方法用于将GRU层的参数进行扁平化处理。在异步训练过程中，PyTorch可能会对模型进行并行处理，这可能导致参数在不同设备上的初始化不一致，从而引入梯度消失的问题。通过扁平化参数，可以解决这个问题
        self.dense = nn.Linear(vision_dim,vision_dim)
        self.layernorm = torch.nn.LayerNorm(vision_dim, eps=1e-12)
        self.criterion = LabelSmoothingCrossEntropy(reduction='sum')
        
#         self._init_weights(self.projection)
#         self._init_weights(self.dense)
#         self._init_weights(self.layernorm)
#         self._init_weights(self.logits[0])
#         self._init_weights(self.logits[1])
        
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.bertconfig.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
        
    def forward(self, inputs, inference=False):
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        qimg_emb = self.vision_encoder(inputs['qImg_batch']) #(B,197,dim)  # 图像端的特征
        qcap_emb = self.text_encoder(input_ids=inputs['qCap_batch'],   # 证据
                                   attention_mask = inputs['qCapmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        ocr_emb = self.text_encoder(input_ids=inputs['ocr_batch'],  # orc文本特征向量
                                   attention_mask = inputs['ocrmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        cap_emb = self.text_encoder(input_ids=inputs['cap_batch'],   # 标题文本特征向量  [B,S,E(768)]
                                   attention_mask = inputs['capmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        
        # image process
        qimg_emb1,_ = self.encoder(qimg_emb)  # 经过GRU，双向GRU,隐藏层大小为原来的一半，双向长度不变
        qimg_emb1 = self.dense(qimg_emb1)  # 线性层
        qimg_emb = self.layernorm(qimg_emb + qimg_emb1)  # 激活层, 在残差之后激活.
        qimg_emb = self.projection(qimg_emb)  # 将长度映射到文本特征向量的长度 512->768
        qimg_mask = torch.ones((qimg_emb.shape[:2])).to(qimg_emb) # 获取图像特征前两个维度的掩码矩阵,就是B*S(S就是批块),实际上图片无须掩码
        
        qimg_emb = qimg_emb / qimg_emb.norm(dim=-1, keepdim=True)  # TODO 每个embeddding内部进行归一化
        qcap_emb = qcap_emb / qcap_emb.norm(dim=-1, keepdim=True)
        cap_emb = cap_emb / cap_emb.norm(dim=-1, keepdim=True)
        ocr_emb = ocr_emb / ocr_emb.norm(dim=-1, keepdim=True)
        
        # 交叉注意力机制, 将图像和文本的特征进行拼接,图像文字特征作为图片特征拼接,文字特征进行拼接
#         text_features, image_features = self.cross_encoder(torch.cat((qcap_emb, cap_emb, ocr_emb), dim=1),
#                         torch.cat((inputs['qCapmask_batch'], inputs['capmask_batch'], inputs['ocrmask_batch']), dim=1), qimg_emb, qimg_mask)
        text_features, image_features = self.cross_encoder(torch.cat((qcap_emb, cap_emb), dim=1),
                        torch.cat((inputs['qCapmask_batch'], inputs['capmask_batch']), dim=1), torch.cat((qimg_emb, ocr_emb),dim=1), torch.cat((qimg_mask, inputs['ocrmask_batch']),dim=1))
#         layers = [self.meanpooling(text_features).squeeze(1),
#                               self.meanpooling(image_features).squeeze(1)]
        # text_features = self.mid_linear(text_features)
        # text_features = self.mid_linear(text_features)
        layers = [text_features[:, 0, :],  
                              image_features[:, 0, :]]  # TODO 获取cls向量,分类头. 经过交叉注意力层之后, 数据长度这个维度发生变化。
        
        prediction = torch.mean(torch.stack([e(layers[i]) for i, e in enumerate(self.logits)]),dim=0)  # 两个特征向量分别作线性变化，然后拼接。TODO stack就是将两个矩阵在最外层添加一个维度，拼接到一起
        # 测试mean
        if inference:
            return prediction #torch.argmax(prediction, dim=1)
        else:
            if self.loss_name=='ce':
                return self.cal_loss(prediction, inputs['labels'])
            else:
                label = inputs['labels'].squeeze(dim=1)
                loss = self.criterion(prediction, label)
                with torch.no_grad():
                    pred_label_id = torch.argmax(prediction, dim=1)
                    accuracy = (label == pred_label_id).float().sum() / label.shape[0]
                return loss, accuracy, pred_label_id, label


    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    def mask_mean(self, x, mask):
        B,L = mask.size()
        mask_x = x*(mask.view(B,L,1,1))
        x_sum = torch.sum(mask_x,dim=1)
        re_x = torch.div(x_sum,torch.sum(mask,dim=1).view(B,1,1))
        return re_x


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        log_pred = torch.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_pred.sum()
        else:
            loss = -log_pred.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()

        return loss * self.eps / c + (1 - self.eps) * torch.nn.functional.nll_loss(log_pred, target,
                                                                                   reduction=self.reduction,
                                                                                   ignore_index=self.ignore_index)

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
    

class RumorDetectionModel_All_Imgs(nn.Module):  # 模型这块
    def __init__(self,                 
                 args,
                lang,
                loss_name='ce'):
        super().__init__() 
        self.loss_name = loss_name  # 损失函数的名字
        if lang =='zh':
            self.vision_encoder = vit(args.vision_backbone_zh)  # clip-vit
#             self.vision_encoder = vlmo_base_patch16(args.vision_backbone_zh)
            self.text_encoder = AutoModel.from_pretrained(args.bert_dir_zh)  # roberta
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_zh+"/config.json")  # 加载roberta的配置文件
        else:
            self.vision_encoder = vit(args.vision_backbone_en)  # clip-vit
            self.text_encoder = AutoModel.from_pretrained(args.bert_dir_en)  # deberta 
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_en+"/config.json")
        self.cross_encoder = LXRTEncoder(self.bertconfig)  # TODO 交叉注意力机制    v
        self.fusion_num = 2  
        self.meanpooling = nn.AdaptiveAvgPool2d((1,self.text_encoder.config.hidden_size))
        # self.mid_linear = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,self.text_encoder.config.hidden_size//2),
        #                                 nn.Dropout(0.1))
        self.logits = nn.ModuleList([nn.Linear(self.text_encoder.config.hidden_size, 2) for i in range(self.fusion_num)])
        # self.logits = nn.Linear(self.text_encoder.config.hidden_size, len(CATEGORY_ID_LIST))
        vision_dim = 512
        self.projection = nn.Sequential(nn.Linear(vision_dim,self.text_encoder.config.hidden_size))
        self.encoder = nn.GRU(input_size=vision_dim, hidden_size=vision_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.encoder.flatten_parameters() # 这个方法用于将GRU层的参数进行扁平化处理。在异步训练过程中，PyTorch可能会对模型进行并行处理，这可能导致参数在不同设备上的初始化不一致，从而引入梯度消失的问题。通过扁平化参数，可以解决这个问题
        self.dense = nn.Linear(vision_dim,vision_dim)
        self.layernorm = torch.nn.LayerNorm(vision_dim, eps=1e-12)
        self.criterion = LabelSmoothingCrossEntropy(reduction='sum')
        
#         self._init_weights(self.projection)
#         self._init_weights(self.dense)
#         self._init_weights(self.layernorm)
#         self._init_weights(self.logits[0])
#         self._init_weights(self.logits[1])
        
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             module.weight.data.normal_(mean=0.0, std=self.bertconfig.initializer_range)
#             if module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
        
    def forward(self, inputs, inference=False):
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        qimg_emb = self.vision_encoder(inputs['imgs_batch']) #(B,197,dim)  # 所有图像端的特征
        qimg_emb = qimg_emb[:, :, 1, :] # 获取cls

        # qimg_emb = self.vision_encoder(inputs['qImg_batch']) #(B,197,dim)  # 图像端的特征
        qcap_emb = self.text_encoder(input_ids=inputs['qCap_batch'],   # 证据
                                   attention_mask = inputs['qCapmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        ocr_emb = self.text_encoder(input_ids=inputs['ocr_batch'],  # orc文本特征向量
                                   attention_mask = inputs['ocrmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        cap_emb = self.text_encoder(input_ids=inputs['cap_batch'],   # 标题文本特征向量  [B,S,E(768)]
                                   attention_mask = inputs['capmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        
        # image process
        qimg_emb1,_ = self.encoder(qimg_emb)  # 经过GRU，双向GRU,隐藏层大小为原来的一半，双向长度不变
        qimg_emb1 = self.dense(qimg_emb1)  # 线性层
        qimg_emb = self.layernorm(qimg_emb + qimg_emb1)  # 激活层, 在残差之后激活.
        qimg_emb = self.projection(qimg_emb)  # 将长度映射到文本特征向量的长度 512->768
        qimg_mask = torch.ones((qimg_emb.shape[:2])).to(qimg_emb) # 获取图像特征前两个维度的掩码矩阵,就是B*S(S就是批块),实际上图片无须掩码
        
        qimg_emb = qimg_emb / qimg_emb.norm(dim=-1, keepdim=True)  # TODO 每个embeddding内部进行归一化
        qcap_emb = qcap_emb / qcap_emb.norm(dim=-1, keepdim=True)
        cap_emb = cap_emb / cap_emb.norm(dim=-1, keepdim=True)
        ocr_emb = ocr_emb / ocr_emb.norm(dim=-1, keepdim=True)
        
        # 交叉注意力机制, 将图像和文本的特征进行拼接,图像文字特征作为图片特征拼接,文字特征进行拼接
#         text_features, image_features = self.cross_encoder(torch.cat((qcap_emb, cap_emb, ocr_emb), dim=1),
#                         torch.cat((inputs['qCapmask_batch'], inputs['capmask_batch'], inputs['ocrmask_batch']), dim=1), qimg_emb, qimg_mask)
        text_features, image_features = self.cross_encoder(torch.cat((qcap_emb, cap_emb), dim=1),
                        torch.cat((inputs['qCapmask_batch'], inputs['capmask_batch']), dim=1), torch.cat((qimg_emb, ocr_emb),dim=1), torch.cat((inputs['imgmask_batch'], inputs['ocrmask_batch']),dim=1))
#         layers = [self.meanpooling(text_features).squeeze(1),
#                               self.meanpooling(image_features).squeeze(1)]
        # text_features = self.mid_linear(text_features)
        # text_features = self.mid_linear(text_features)
        layers = [text_features[:, 0, :],  
                              image_features[:, 0, :]]  # TODO 获取cls向量,分类头. 经过交叉注意力层之后, 数据长度这个维度发生变化。
        
        prediction = torch.mean(torch.stack([e(layers[i]) for i, e in enumerate(self.logits)]),dim=0)  # 两个特征向量分别作线性变化，然后拼接。TODO stack就是将两个矩阵在最外层添加一个维度，拼接到一起
        # 测试mean
        if inference:
            return prediction #torch.argmax(prediction, dim=1)
        else:
            if self.loss_name=='ce':
                return self.cal_loss(prediction, inputs['labels'])
            else:
                label = inputs['labels'].squeeze(dim=1)
                loss = self.criterion(prediction, label)
                with torch.no_grad():
                    pred_label_id = torch.argmax(prediction, dim=1)
                    accuracy = (label == pred_label_id).float().sum() / label.shape[0]
                return loss, accuracy, pred_label_id, label


    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    def mask_mean(self, x, mask):
        B,L = mask.size()
        mask_x = x*(mask.view(B,L,1,1))
        x_sum = torch.sum(mask_x,dim=1)
        re_x = torch.div(x_sum,torch.sum(mask,dim=1).view(B,1,1))
        return re_x


class RumorDetectionModel_en(nn.Module):
    def __init__(self,                 
                 args,
                lang):
        super().__init__() 
        
        if lang =='zh':
            self.vision_encoder = vit(args.vision_backbone_zh)
            self.text_encoder = AutoModel.from_pretrained(args.bert_dir_zh) 
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_zh+"/config.json")
        elif lang == 'en':
            self.vision_encoder = vit(args.vision_backbone_en)
            self.text_encoder = AutoModel.from_pretrained(args.bert_dir_en) 
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_en+"/config.json")
        else:
            self.vision_encoder = vit(args.vision_backbone_en)
            self.text_encoder = AutoModel.from_pretrained(args.roberta) 
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_en+"/config.json")
        self.cross_encoder = LXRTEncoder(self.bertconfig)
        self.fusion_num = 2
        self.meanpooling = nn.AdaptiveAvgPool2d((1,self.text_encoder.config.hidden_size))
        self.logits = nn.ModuleList([nn.Linear(self.text_encoder.config.hidden_size, 3) for i in range(self.fusion_num)])
        # self.logits = nn.Linear(self.text_encoder.config.hidden_size, len(CATEGORY_ID_LIST))
        vision_dim = 512
        self.projection = nn.Sequential(nn.Linear(vision_dim,self.text_encoder.config.hidden_size))
        self.encoder = nn.GRU(input_size=vision_dim,hidden_size=vision_dim//2,num_layers=1,batch_first=True,bidirectional=True)
        self.encoder.flatten_parameters()
        self.dense = nn.Linear(vision_dim,vision_dim)
        self.layernorm = torch.nn.LayerNorm(vision_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, inputs, inference=False):
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        qimg_emb = self.vision_encoder(inputs['qImg_batch']) #(B,197,dim)
        qcap_emb = self.text_encoder(input_ids=inputs['qCap_batch'], 
                                   attention_mask = inputs['qCapmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        ocr_emb = self.text_encoder(input_ids=inputs['ocr_batch'], 
                                   attention_mask = inputs['ocrmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        cap_emb = self.text_encoder(input_ids=inputs['cap_batch'], 
                                   attention_mask = inputs['capmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        
        # image process
        qimg_emb1,_ = self.encoder(qimg_emb)
        qimg_emb1 = self.dense(qimg_emb1)
        qimg_emb = self.layernorm(qimg_emb + qimg_emb1)
        qimg_emb = self.projection(qimg_emb)
        qimg_mask = torch.ones((qimg_emb.shape[:2])).to(qimg_emb)
        
        qimg_emb = qimg_emb / qimg_emb.norm(dim=-1, keepdim=True)
        qcap_emb = qcap_emb / qcap_emb.norm(dim=-1, keepdim=True)
        cap_emb = cap_emb / cap_emb.norm(dim=-1, keepdim=True)
        ocr_emb = ocr_emb / ocr_emb.norm(dim=-1, keepdim=True)
        
#         text_features, image_features = self.cross_encoder(torch.cat((qcap_emb, cap_emb, ocr_emb), dim=1),
#                         torch.cat((inputs['qCapmask_batch'], inputs['capmask_batch'], inputs['ocrmask_batch']), dim=1), qimg_emb, qimg_mask)
        text_features, image_features = self.cross_encoder(torch.cat((qcap_emb, cap_emb), dim=1),
                        torch.cat((inputs['qCapmask_batch'], inputs['capmask_batch']), dim=1), torch.cat((qimg_emb, ocr_emb),dim=1), torch.cat((qimg_mask, inputs['ocrmask_batch']),dim=1))
#         layers = [self.meanpooling(text_features).squeeze(1),
#                               self.meanpooling(image_features).squeeze(1)]
        layers = [text_features[:,0,:],
                              image_features[:,0,:]]
        
        prediction = torch.mean(torch.stack([e(layers[i]) for i, e in enumerate(self.logits)]),dim=0)
        # 测试mean
        if inference:
            return prediction #torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['labels'])
                                  
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label,weight = torch.tensor([0.2,0.4,0.4]).to('cuda'))
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    def mask_mean(self, x, mask):
        B,L = mask.size()
        mask_x = x*(mask.view(B,L,1,1))
        x_sum = torch.sum(mask_x,dim=1)
        re_x = torch.div(x_sum,torch.sum(mask,dim=1).view(B,1,1))
        return re_x

class RumorDetectionModel_no_ocr(nn.Module):
    def __init__(self,                 
                 args,
                lang):
        super().__init__() 
        
        if lang =='zh':
            self.vision_encoder = vit(args.vision_backbone_zh)
            self.text_encoder = AutoModel.from_pretrained(args.bert_dir_zh) 
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_zh+"/config.json")
        elif lang == 'en':
            self.vision_encoder = vit(args.vision_backbone_en)
            self.text_encoder = AutoModel.from_pretrained(args.bert_dir_en) 
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_en+"/config.json")
        else:
            self.vision_encoder = vit(args.vision_backbone_en)
            self.text_encoder = AutoModel.from_pretrained(args.roberta) 
            # self.text_encoder = XLMRModel.from_pretrained(args.bert_dir,checkpoint_file='model.pt')   
            self.bertconfig = BertConfig.from_json_file(args.bert_dir_en+"/config.json")
        self.cross_encoder = LXRTEncoder(self.bertconfig)
        self.fusion_num = 2
        self.meanpooling = nn.AdaptiveAvgPool2d((1,self.text_encoder.config.hidden_size))
        self.logits = nn.ModuleList([nn.Linear(self.text_encoder.config.hidden_size, 3) for i in range(self.fusion_num)])
        # self.logits = nn.Linear(self.text_encoder.config.hidden_size, len(CATEGORY_ID_LIST))
        vision_dim = 512
        self.projection = nn.Sequential(nn.Linear(vision_dim,self.text_encoder.config.hidden_size))
        self.encoder = nn.GRU(input_size=vision_dim,hidden_size=vision_dim//2,num_layers=1,batch_first=True,bidirectional=True)
        self.encoder.flatten_parameters()
        self.dense = nn.Linear(vision_dim,vision_dim)
        self.layernorm = torch.nn.LayerNorm(vision_dim, eps=1e-12)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, inputs, inference=False):
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda()
        qimg_emb = self.vision_encoder(inputs['qImg_batch']) #(B,197,dim)
        qcap_emb = self.text_encoder(input_ids=inputs['qCap_batch'], 
                                   attention_mask = inputs['qCapmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        ocr_emb = self.text_encoder(input_ids=inputs['ocr_batch'], 
                                   attention_mask = inputs['ocrmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        cap_emb = self.text_encoder(input_ids=inputs['cap_batch'], 
                                   attention_mask = inputs['capmask_batch'],
                                   output_hidden_states=False,
                                   return_dict = True
                                  ).last_hidden_state
        
        # image process
        qimg_emb1,_ = self.encoder(qimg_emb)
        qimg_emb1 = self.dense(qimg_emb1)
        qimg_emb = self.layernorm(qimg_emb + qimg_emb1)
        qimg_emb = self.projection(qimg_emb)
        qimg_mask = torch.ones((qimg_emb.shape[:2])).to(qimg_emb)
        
        qimg_emb = qimg_emb / qimg_emb.norm(dim=-1, keepdim=True)
        qcap_emb = qcap_emb / qcap_emb.norm(dim=-1, keepdim=True)
        cap_emb = cap_emb / cap_emb.norm(dim=-1, keepdim=True)
        # ocr_emb = ocr_emb / ocr_emb.norm(dim=-1, keepdim=True)
        
#         text_features, image_features = self.cross_encoder(torch.cat((qcap_emb, cap_emb, ocr_emb), dim=1),
#                         torch.cat((inputs['qCapmask_batch'], inputs['capmask_batch'], inputs['ocrmask_batch']), dim=1), qimg_emb, qimg_mask)
        text_features, image_features = self.cross_encoder(torch.cat((qcap_emb, cap_emb), dim=1),
                        torch.cat((inputs['qCapmask_batch'], inputs['capmask_batch']), dim=1), qimg_emb,  qimg_mask)
        
#         layers = [self.meanpooling(text_features).squeeze(1),
#                               self.meanpooling(image_features).squeeze(1)]
        layers = [text_features[:,0,:],
                              image_features[:,0,:]]
        
        prediction = torch.mean(torch.stack([e(layers[i]) for i, e in enumerate(self.logits)]),dim=0)
        # 测试mean
        if inference:
            return prediction #torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['labels'])
                                  
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label,weight = torch.tensor([0.2,0.4,0.4]).to('cuda'))
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label
    def mask_mean(self, x, mask):
        B,L = mask.size()
        mask_x = x*(mask.view(B,L,1,1))
        x_sum = torch.sum(mask_x,dim=1)
        re_x = torch.div(x_sum,torch.sum(mask,dim=1).view(B,1,1))
        return re_x