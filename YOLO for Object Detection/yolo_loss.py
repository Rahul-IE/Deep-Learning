import torch

class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super().__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj
    
    def compute_iou(self, box1, box2):
        """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        """
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)

        return iou

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        box_copy = boxes.clone()
        
        boxes = torch.tensor([(box_copy[0]/float(self.S)) - 0.5*box_copy[2], (box_copy[1]/float(self.S)) - 0.5*box_copy[3], 
                          (box_copy[0]/float(self.S)) + 0.5*box_copy[2], (box_copy[1]/float(self.S)) + 0.5*box_copy[3]])
        
        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : [(tensor) size (-1, 5) ...]
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###

        #always pass box_target as one argument to the compute_iou function
        
        best_iou = torch.zeros(box_target.shape[0], 1)
        best_boxes = torch.zeros(box_target.shape[0], 4)
        best_idxs = torch.zeros(box_target.shape[0], 1)
        
        for cell in range(box_target.shape[0]):
            
            iou_cell_box = []

            for box in pred_box_list:

                box[cell,:] = self.xywh2xyxy(box[cell,:4])
                
                box_target[cell,:4] = self.xywh2xyxy(box_target[cell,:4])

                iou_cell_box.append(self.compute_iou(box[cell,:4].view(1,-1), box_target[cell,:4].view(1,-1)))
            
            best_iou[cell,0] = max(iou_cell_box).item()
            
            best_boxes[cell,:] = pred_box_list[np.argmax([i.item() for i in iou_cell_box])][cell,:]
            
            best_idxs[cell,:] = np.argmax([i.item() for i in iou_cell_box])

        return best_iou, best_boxes, best_idxs

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        coord_mask = has_object_map == True
        coord_mask = coord_mask.unsqueeze(-1).expand_as(classes_pred)
        
        class_pred = classes_pred[coord_mask].view(-1, 20)
        
        class_target = classes_target[coord_mask].view(-1, 20)
        
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')
        
        return loss_class

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        
        pred_tensor_list = pred_boxes_list
        
        noobj_mask = has_object_map == False
        pred_box = pred_tensor_list[0]

        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(pred_box).cuda()
        noobj_pred = torch.cat((pred_tensor_list[0][noobj_mask].view(-1, pred_box.size(3)).cuda(),
                                pred_tensor_list[1][noobj_mask].view(-1, pred_box.size(3)).cuda()), 1)

        noobj_conf_mask = torch.zeros(noobj_pred.shape).cuda() 
        noobj_conf_mask[:, 4] = 1
        noobj_conf_mask[:, 9] = 1

        noobj_pred_conf = noobj_pred[noobj_conf_mask].view(-1, 2)
        
        noobj_target_conf = torch.zeros(noobj_pred_conf.shape).cuda()
        print(noobj_target_conf.requires_grad)
        
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        
        return loss_noobj
      
    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        
        loss = F.mse_loss(box_pred_conf, box_target_conf.detach(), reduction='sum')
        
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """

        reg_loss = F.mse_loss(box_pred_response[:,:2].cuda(), box_target_response[:,:2].cuda(), reduction='sum')     
        + F.mse_loss(torch.sqrt(box_pred_response[:,2:4].cuda()), torch.sqrt(box_target_response[:,2:4].cuda()), reduction='sum')

        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) N:batch_size
                      where B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        pred_tensor_list = [pred_tensor[:,:,:, :5] , pred_tensor[:,:,:, 5:10]]
        pred_boxes_list = pred_tensor_list
        
        # -- pred_cls (containing all classification prediction)
        pred_cls = pred_tensor[:,:,:, 10:30] 
        
        # compcute classification loss
        class_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)/N
        
        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)/N
        
        coord_mask = has_object_map == True 
        filter_pred = coord_mask.unsqueeze(-1).expand_as(pred_tensor) 
        coord_pred = pred_tensor[filter_pred].view(-1, pred_tensor.shape[3])
        pred_cls = pred_tensor[:,:,:, 10:30]
       
        pred_boxes_list[0] = coord_pred[:, :5].contiguous().view(-1, 5)
        pred_boxes_list[1] = coord_pred[:, 5:10].contiguous().view(-1, 5)
        
        filter_target = coord_mask.unsqueeze(-1).expand_as(target_boxes)
        target_boxes = target_boxes[filter_target].view(-1,4)
        
        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        pred_box_list = [pred_boxes_list[0][:,:4], pred_boxes_list[1][:,:4]]
        best_ious, best_boxes, best_idxs = self.find_best_iou_boxes(pred_box_list, target_boxes) 
        
        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        box_pred_response = best_boxes[:, :4]
        box_target_response = target_boxes[:, :4]
        reg_loss = self.get_regression_loss(box_pred_response, box_target_response)/N
        
        # compute contain_object_loss
        conf_idxs = torch.where(best_idxs == 0, 4, 9).view(-1,).tolist()
        row_idxs = [i for i in range(target_boxes.shape[0])]
        
        box_pred_conf = coord_pred[row_idxs, conf_idxs].view(-1,1).cuda()
        box_target_conf = best_ious.cuda()

        cont_obj_loss = self.get_contain_conf_loss(box_pred_conf, box_target_conf)/N
        
        # compute final loss
        final_loss = class_loss + no_obj_loss + reg_loss + cont_obj_loss
        
        # construct return loss_dict
        loss_dict = dict(total_loss = final_loss, reg_loss = reg_loss, containing_obj_loss = cont_obj_loss, no_obj_loss = no_obj_loss,
            cls_loss = class_loss)
        
        return loss_dict