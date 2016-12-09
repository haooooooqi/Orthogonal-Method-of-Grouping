kSparse, Parent = torch.class('nn.kSparse', 'nn.Module')

--[[
function kSparse:__init(k1, k2)
    Parent.__init(self)
    self.kSpatial = k1 or 1 -- top k value keep in each m * m image
    self.kFilter = k2 or 1 -- top k value keep in each m * m image
    self.count = 30
end

--]]

function kSparse:__init(k1, k2)
    Parent.__init(self)
    self.kSpatial = k1 or 1 -- top k value keep in each m * m image
    self.kFilter = k2 or 1 -- top k value keep in each m * m image
    self.count = 32
    self.count_ = 1
    self.isOpen = true
end


--[[
function kSparse:updateOutput(input)
    self.maxPos = input:clone():fill(0)
    self.output = input:clone()
    --self.output:resizeAs(input):copy(input) --suppose 10 * 32 * 64 * 64
    for i = 1, self.output:size()[1] do -- 32 * 64 * 64
        local filterMax = torch.Tensor(self.output:size()[2])
        --Spatial Sparsity
        for j = 1, self.output:size()[2] do -- 64 * 64
            local tmpOutput = self.output[{{i},{j}}]
            local tmpMax = self.maxPos[{{i},{j}}]:resize(self.output:size()[3] ^ 2)
            _, ind = tmpOutput:resize(self.output:size()[3] ^ 2):topk(self.kSpatial, true)
            local copyMax = tmpOutput:index(1, ind):clone()
            tmpOutput:fill(0)
            for k = 1, ind:size()[1] do
                tmpMax[ind[k] ] = 1
                tmpOutput[ind[k] ] = copyMax[k]
            end
            filterMax[j] = copyMax:max()
        end
        --Filter Sparsity
        if self.output:size()[2] - self.kFilter > 0 then
            _, ind = filterMax:topk(self.output:size()[2] - self.kFilter)
            for j = 1, ind:size()[1] do
                self.output[i][ind[j] ]:fill(0)
                self.maxPos[i][ind[j] ]:fill(0)
            end
        end
    end
    return self.output
end
--]]


--[[


function kSparse:updateOutput(input)
    self.maxPos = input:clone():fill(0)
    self.output = input:clone()
    self.count = self.count - 1
    if self.kSpatial ==0 or self.kFilter == 0 then
        self.output:fill(0)
        return self.output:cuda()
    end
    --local kSpatial = math.max(math.min(self.kSpatial + self.count, input:size()[2]), self.kSpatial)
    local kSpatial = self.kSpatial
    local kFilter = math.max(math.min(self.kFilter + self.count, input:size()[2]), self.kFilter)
    --print(kSpatial, kFilter)
    --self.output:resizeAs(input):copy(input) --suppose 10 * 32 * 64 * 64
    for i = 1, self.output:size()[1] do -- 32 * 64 * 64
        local filterMax = torch.Tensor(self.output:size()[2]):float()
        --Spatial Sparsity
        for j = 1, self.output:size()[2] do -- 64 * 64
            local tmpOutput_ = self.output[{{i},{j}}]:resize(self.output:size()[3] ^ 2)
            local tmpMax = self.maxPos[{{i},{j}}]:resize(self.output:size()[3] ^ 2)
            local tmpOutput = self.output[{{i},{j}}]:resize(self.output:size()[3] ^ 2):float()
            --local tmpMax = self.maxPos[{{i},{j}}]:resize(self.output:size()[3] ^ 2):float()
            --_, ind_ = tmpOutput:topk(self.kSpatial, true)
            _, ind_ = torch.topk(tmpOutput, kSpatial, 1, true)
            local copyMax = tmpOutput:index(1, ind_):clone()
            tmpOutput_:fill(0)
            for k = 1, ind_:size()[1] do
                --local index = math.floor(ind_[k])
                local index = ind_[k]
                if index % 1 ~= 0 or index <= 0 or index > tmpMax:size()[1] then
                    print(tmpMax:size())
                    print(index)
                    print(ind_)
                end
                tmpMax[index] = 1
                tmpOutput_[index] = copyMax[k]
            end
            filterMax[j] = copyMax:max()
        end
        --Filter Sparsity
        if self.output:size()[2] - kFilter > 0 then
            --_, ind = filterMax:topk(self.output:size()[2] - self.kFilter)
            _, ind = torch.topk(filterMax, self.output:size()[2] - kFilter)
            for j = 1, ind:size()[1] do
                local index = ind[j]
                self.output[i][index]:fill(0)
                self.maxPos[i][index]:fill(0)
            end
        end
    end
    return self.output:cuda()
end

--]]




function kSparse:openGate()
    self.isOpen = true
end

function kSparse:closeGate()
    self.isOpen = false
end

function kSparse:getMaxActivatedFilter()
    return self.maxActivatedFilter
end


function kSparse:updateOutput(input)
    self.maxActivatedFilter = torch.Tensor(input:size()[1], input:size()[2])
    self.maxPos = input:clone():fill(0)
    self.output = input:clone()
    self.count_ = self.count_ + 1
    if self.count_ > 50 then
        self.count = self.count - 1
        self.count_ = 0
    end
    
    if self.isOpen == false or self.kSpatial == 0 or self.kFilter == 0 then
        print('kSparse layer been closed')
        self.output:fill(0)
        return self.output:cuda()
    end
    --local kSpatial = math.max(math.min(self.kSpatial + self.count, input:size()[2]), self.kSpatial)
    local kSpatial = self.kSpatial
    local kFilter = math.max(math.min(self.kFilter + self.count, input:size()[2]), self.kFilter)
    --print(kSpatial, kFilter)
    --self.output:resizeAs(input):copy(input) --suppose 10 * 32 * 64 * 64
    for i = 1, self.output:size()[1] do -- 32 * 64 * 64
        local filterMax = torch.Tensor(self.output:size()[2]):float()
        --Spatial Sparsity
        for j = 1, self.output:size()[2] do -- 64 * 64
            local tmpOutput_ = self.output[{{i},{j}}]:resize(self.output:size()[3] * self.output:size()[4])
            local tmpMax = self.maxPos[{{i},{j}}]:resize(self.output:size()[3] * self.output:size()[4])
            local tmpOutput = self.output[{{i},{j}}]:resize(self.output:size()[3] * self.output:size()[4]):float()
            --local tmpMax = self.maxPos[{{i},{j}}]:resize(self.output:size()[3] ^ 2):float()
            --_, ind_ = tmpOutput:topk(self.kSpatial, true)
            _, ind_ = torch.topk(tmpOutput, kSpatial, 1, true)
            local copyMax = tmpOutput:index(1, ind_):clone()
            tmpOutput_:fill(0)
            for k = 1, ind_:size()[1] do
                --local index = math.floor(ind_[k])
                local index = ind_[k]
                if index % 1 ~= 0 or index <= 0 or index > tmpMax:size()[1] then
                    print(tmpMax:size())
                    print(index)
                    print(ind_)
                end
                tmpMax[index] = 1
                tmpOutput_[index] = copyMax[k]
            end
            filterMax[j] = copyMax:max()
        end
        self.maxActivatedFilter[i] = filterMax
        --Filter Sparsity
        if self.output:size()[2] - kFilter > 0 then
            --_, ind = filterMax:topk(self.output:size()[2] - self.kFilter)
            _, ind = torch.topk(filterMax, self.output:size()[2] - kFilter)
            for j = 1, ind:size()[1] do
                local index = ind[j]
                self.output[i][index]:fill(0)
                self.maxPos[i][index]:fill(0)
            end
        end
    end
    if self.isOpen == false then
        self.maxPos:fill(0)
    end
    
    --print(self.maxActivatedFilter)
    return self.output:cuda()
end





function kSparse:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   self.gradInput:cmul(self.maxPos) -- simply mask the gradients with the noise vector
   return self.gradInput
end

print('model:add(nn.kSparse(1, 16))')
