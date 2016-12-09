require 'nn'
require 'nngraph'


local layer, parent = torch.class('nn.pass', 'nn.Module')

function layer:__init(opt)
    self.hSize = opt.hSize
    self.ifTrain = opt.ifTrain
    self.model = nn.Sequential()
            :add(nn.CMul(opt.hSize))
            :add(nn.Add(opt.hSize))
    self.model:get(1).weight:fill(1)
    self.model:get(2).bias:fill(0)
    self.weights, self.gradWeights = self.model:getParameters()

end

function layer:parameters()
    return self.weights, self.gradWeights
end

function layer:getParameters()
    return self.weights, self.gradWeights
end

function layer:evaluate()
    --do nothing
end

function layer:updateOutput(input)
    return self.model:forward(input)
end

function layer:updateGradInput(input, gradOutput)
    return self.model:backward(input, gradOutput)
end

function setGroup(dict)
    self.gSize = #dict
    self.group = dict
end


function layer:accGradParameters(input, gradOutput, scale)
    if self.ifTrain == false then
        self.gradWeights:fill(0)
    else
        for dim = 1, 2 do
            local tempW = self.weights[{{(dim - 1) * self.hSize + 1, dim * self.hSize}}]
            for i = 1, #self.group do
                local index = torch.LongTensor(self.group[i])
                local mean = tempW:index(1, index):mean()
                tempW:scatter(1, index, mean)
            end
            self.gradWdiths[{{(dim - 1) * self.hSize + 1, dim * self.hSize}}] = tempW
        end
    end
end

function layer:open()
    self.ifTrain = true
end

function layer:close()
    self.ifTrain = false
end
