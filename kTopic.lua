kTopic, Parent = torch.class('nn.kTopic', 'nn.Module')

function kTopic:__init(k1, k2, filterNum)
    Parent.__init(self)
    self.kInner = k1 or 1 -- kInner number of activated values in the topic
    self.kInter = k2 or 1 -- Select kInter most activated topics
    self.filterNumPerTopic = filterNum
    self.count = 3
    self.count_ = 1
    self.isOpen = true
end

function kTopic:openGate()
    self.isOpen = true
end

function kTopic:closeGate()
    self.isOpen = false
end
--[[
function kTopic:getMaxActivatedFilter()
    return self.maxActivatedFilter
end
]]--
function kTopic:updateOutput(input)
    self.output = input:clone()
    self.maxPos = input:clone():fill(1)
    self.count_ = self.count_ + 1
    if self.count_ >= 50 then
        self.count = self.count - 1
        self.count_ = 0
    end
    if self.isOpen == false or self.kSpatial == 0 or self.kFilter == 0 then
        print('kTopic layer has been blocked')
        self.output:fill(0)
        self.maxPos:fill(0)
        return self.output:cuda()
    end
    
    
    local kInner = self.kInner
    local kInter = math.max(self.kInter, math.min(self.count, self.output:size()[2] / self.filterNumPerTopic))
    for i = 1, self.output:size()[1] do -- [10] * topic * filters * 64 * 64
        local filterMax = torch.Tensor(self.output:size()[2]):float()
        -- Find the largest activated value for each filter map
        for j = 1, self.output:size()[2] do -- [10] * [topic * filters] * 64 * 64
            local tmpOutput_ = self.output[{{i}, {j}}]:resize(self.output:size()[3] * self.output:size()[4]):float()
            --val_, _ = torch.topk(tmpOutput_, kInner, true)
            filterMax[j] = tmpOutput_:max()
        end
        -- reshape the max to topic * filters/ topic
        filterMax = filterMax:reshape(input:size()[2] / self.filterNumPerTopic, self.filterNumPerTopic)
        -- Get the topk activate per topic
        val_, _ = torch.topk(filterMax, kInner, true)
        -- Calculate the mean value of the top k activate for each topic
        local meanVal = val_:mean(2)
        meanVal = meanVal:reshape(meanVal:size()[1])
        -- Select the top k(self.Inter) topic as output
        if input:size()[2] / self.filterNumPerTopic - kInter > 0 then
            _, ind_ = torch.topk(meanVal, input:size()[2] / self.filterNumPerTopic - kInter)
            for iter  = 1, ind_:size()[1] do
                local iterInd = ind_[iter]
                self.maxPos[{{i}, {(iterInd - 1) * self.filterNumPerTopic + 1, iterInd * self.filterNumPerTopic}, {}}]:fill(0)
                self.output[{{i}, {(iterInd - 1) * self.filterNumPerTopic + 1, iterInd * self.filterNumPerTopic}, {}}]:fill(0)
            end
        end
    end
    --print(self.maxPos)
    return self.output:cuda()
end





function kTopic:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.maxPos) -- simply mask the gradients with the noise vector
   return self.gradInput
end

print('model:add(nn.kTopic(kInner, kInter, filterNum)')
