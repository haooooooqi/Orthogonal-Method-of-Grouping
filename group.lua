require 'nn';
require 'nngraph';
require 'math';
--debugger = require 'fb.debugger'


local layer, parent = torch.class('nn.group', 'nn.Module')

function layer:__init(opt)
    self.inDim = opt.inDim
    self.outDim = opt.outDim
    self.kSize = opt.kSize
    self.gSize = opt.gSize
    self.model = nn.SpatialConvolutionMM(self.inDim, self.outDim, self.kSize, self.kSize)
    self.weights, self.gradWeights = self.model:getParameters()
    self.pool = nn.SpatialMaxPooling(5, 5, 3, 3)
    self.lambda = -0.00001
    self.beta = 0.00001
    self.beta = 0.0
    self.lambda = 0.0000
    self.mulTable = nn.CMulTable()
    self.weight = self.model.weight

    self.dW = self.model.dW
    self.dH = self.model.dH
    self.padW = self.model.padW
    self.padH = self.model.padH
    self.bias = self.model.bias
    self.gradWeight = self.model.gradWeight
    self.gradBias = self.model.gradBias


    --Initialize group setting
    self.group = {}
    local tempGroup = {}
    local tempSize = math.floor(self.outDim / self.gSize)
    for i = 1, self.outDim do
        table.insert(tempGroup, i)
        if i % tempSize == 0 and
            i / tempSize < self.gSize then
            table.insert(self.group, tempGroup)
            tempGroup = {}
        end
    end
    if #tempGroup > 0 then
        table.insert(self.group, tempGroup)
    end
end

function layer:parameters()
    return self.model:parameters()
end

function layer:getParameters()
    return self.model:getParameters()
end

function layer:updateGroup()
    local minVal = 1e9
    local minID = -1
    local oriGroup = -1
    local weights = self.model.weight
    for i = 1, #self.group do
        if #self.group[i] > 1 then
            local index = torch.LongTensor(self.group[i])
            local matrix = weights:index(1, index):clone():reshape(#self.group[i], self.kSize * self.kSize)
            local dis = torch.mm(
                    matrix[{{2, #self.group[i]}}],
                    matrix[{{1}, {}}]:transpose(1, 2)
                )
            local val, ind = dis:reshape(#self.group[i] - 1):min(1)
            val = val[1]
            ind = ind[1]
            if val < minVal then
                minID = index[ind + 1]
                minVal = val
                oriGroup = i
            end
        end
    end
    local maxVal = -1e9
    local maxID = -1
    local weight = weights[{{minID}, {}}]
    weight = weight:reshape(weight:size(1), self.kSize * self.kSize)
    for i = 1, #self.group do
        if #self.group[i] >= 1 then
            local temp = weights[{{self.group[i][1]}, {}}]
            temp = temp:reshape(temp:size(1), self.kSize * self.kSize)
            local dis = torch.mm(
                    weight,
                    temp:transpose(1, 2)
                )[1]
            if dis[1] > maxVal then
                maxID = i
                maxVal = dis[1]
            end
        end
    end
    for key, value in pairs(self.group[oriGroup]) do
        if value == minID then
            table.remove(self.group[oriGroup], key)
        end
    end
    table.insert(self.group[maxID], minID)
    for i = 1, #self.group do
        local tempPerm = torch.randperm(#self.group[i])
        for j = 1, #self.group[i] do
            local ind = tempPerm[j]
            self.group[i][j], self.group[i][ind] = self.group[i][ind], self.group[i][j]
        end
        --local ind = math.random(1, #self.group[i])
        --self.group[i][1], self.group[i][ind] = self.group[i][ind], self.group[i][1]
    end
end
            

function layer:lossInner()
    local midState = self.poolOutput
    local res = midState:clone():fill(0)
    for i = 1, #self.group do
        local first = self.group[i][1]
        for j = 2, #self.group[i] do
            local second = self.group[i][j]
            res[{{}, {second}, {}}] = 
                midState[{{}, {first}, {}}] - midState[{{}, {second}, {}}]
        end
    end
    res:mul(self.lambda) 
    return res 
end

function layer:lossInter()
    local midState = self.poolOutput
    local tempTable = {}
    for i = 1, #self.group do
        table.insert(tempTable, midState[{{}, {self.group[i][1]}, {}}])
    end
    local out = self.mulTable:forward(tempTable):clone():mul(self.beta)
    out = torch.expand(out, out:size(1), self.outDim, out:size(3), out:size(4))
    return out
end

function layer:evaluate()
    --do nothing
end

function layer:updateOutput(input)
    self.output = self.model:forward(input) -- you might want to try add *sigmoid() here
    self.poolInput = self.output:clone() -- you might want to try add :tanh()
    self.poolOutput = self.pool:forward(self.output):clone()
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    local innerLoss = self:lossInner()
    local interLoss = self:lossInter()
    local gradPool = self.pool:backward(self.poolInput, innerLoss + interLoss):clone()
    local res = self.model:backward(
            input, 
            gradOutput + gradPool
        )
--    self.model.gradWeight:mul(0.2)
    --self.gradWeights:add(torch.sign(self.weights):mul(self.coefL1) + self.weights:clone():mul(self.coefL2))

    self:updateGroup()
    return res
end

function getGroup(dict)
    return self.group
end

function layer:accGradParameters(input, gradOutput, scale)

end

