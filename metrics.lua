--[[
    From Pixels to Torques: Policy Learning using Deep Dynamical Convolutional Networks

    Copyright (C) 2015 John-Alexander M. Assael (www.johnassael.com)

    The MIT License (MIT)

    Permission is hereby granted, free of charge, to any person obtaining a copy of
    this software and associated documentation files (the "Software"), to deal in
    the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
    of the Software, and to permit persons to whom the Software is furnished to do
    so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

]]--

function f1_score(y_pred, y_true)
    local f1_scores = {}
    --local y_pred = y_pred:float()
    local y_true = y_true:float():byte()
    local pred = y_pred:float():gt(0.5):byte()
    
    for i = 1,y_true:size(1) do
        local y = y_true[i]

        local tp = torch.sum(y[pred[i]])
        local fp = torch.sum(pred[i]) - tp
        local fn = torch.sum(y) - tp
        
        local precision, recall
        if tp+fp > 0 then
            precision = tp/(tp+fp)
        else
            precision = 0
        end
        
        if tp+fn > 0 then
            recall = tp/(tp+fn)
        else
            recall = 0
        end
        
        if (precision+recall) > 0 then
            local f1 = 2*precision*recall/(precision+recall)
            f1_scores[#f1_scores+1] = f1
        else
            f1_scores[#f1_scores+1] = 0            
        end
    end
    
    if #f1_scores > 0 then
        return torch.DoubleTensor(f1_scores):mean()
    else
        return 0
    end
end

function accuracy_score(y_pred, y_true)
    local acc_scores = {}
    local y_true = y_true:float():byte()
    local pred = y_pred:float():gt(0.5):byte()
    
    for i = 1,y_true:size(1) do
        local y = y_true[i]
        
        local acc = torch.sum(y_true[i]:eq(pred[i])) / y_true:size(2)
        
        acc_scores[#acc_scores+1] = acc
    end
    
    if #acc_scores > 0 then
        return torch.DoubleTensor(acc_scores):mean()
    else
        return 0
    end
end