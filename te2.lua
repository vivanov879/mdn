require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'

local n_gaussians = 3
local n_features = 5
local n_labels = 2


features = nn.Identity()()

z_mu = nn.Linear(n_features, n_labels * n_gaussians)(features)
z_sigma = nn.Linear(n_features, n_gaussians)(features)
z_alpha = nn.Linear(n_features, n_gaussians)(features)

alpha = nn.SoftMax()(z_alpha)
mu = nn.Reshape(n_labels, n_gaussians)(z_mu)
sigma = nn.Exp()(z_sigma)

labels = nn.Identity()()

l = {}

for i = 1, n_gaussians do
  z1 = nn.Select(3,i)(mu)
  z2 = nn.MulConstant(-1)(z1)
  d = nn.CAddTable()({z2, labels})
  sq = nn.Square()(d)
  s = nn.Sum(2)(sq)
  s = nn.MulConstant(-0.5)(s)
  sigma_select = nn.Select(2,i)(sigma)
  sigma_inv = nn.Power(-1)(sigma_select)
  sigma_sq_inv = nn.Square()(sigma_inv)
  alpha_select = nn.Select(2,i)(alpha)
  mm = nn.CMulTable()({s, sigma_sq_inv, alpha_select})
  e = nn.Exp()(mm)
  r = nn.CMulTable()({e, sigma_inv})
  r = nn.MulConstant(math.pow((2 * math.pi), -0.5))(r)
  l[#l + 1] = r
end

z = nn.CAddTable()(l)
z = nn.Log()(z)
z = nn.MulConstant(-1)(z)

m = nn.gModule({features, labels}, {z, alpha, mu, sigma})


--использовать MSECriterion с target=0, потому что там под логарифмом взвешенная с положительными весами сумма нормальных распределний, что меньше единицы, так как сумма alpha = 1. то есть отрицат логарифм не будет < 0

criterion = nn.MSECriterion()


--заделаем тестовые features и labels
local n_data = 100
local features_input = torch.zeros(n_data, n_features)
local labels_input = torch.zeros(n_data, n_labels)
for i = 1, n_data do 
  element = torch.ones(n_features)

  element:mul(i)
  element[2] = 0.5 * i
  element[3] = 1
  element:add(torch.rand(element:size()))
  features_input[{{i}, {}}] = element
  
  label = torch.ones(n_labels)
  label:mul(2 * i)
  label[1] = 1
  label:add(torch.rand(label:size()))
  labels_input[{{i}, {}}] = label
  
end

--

local model, criterion = m, criterion
local params, grads = model:getParameters()


-- return loss, grad
local feval = function(x)
  if x ~= params then
    params:copy(x)
  end
  grads:zero()

  -- forward
  local outputs, alpha, mu, sigma = unpack(model:forward({features_input, labels_input}))
  local targets = torch.zeros(outputs:size())
  local loss = criterion:forward(outputs, targets)
  -- backward
  local dsigma = torch.zeros(sigma:size())
  local dalpha = torch.zeros(alpha:size())
  local dmu = torch.zeros(mu:size())
  local dloss_doutput = criterion:backward(outputs, targets)
  model:backward(data.inputs, {dloss_doutput, dalpha, dmu, dsigma})

  return loss, grads
end

------------------------------------------------------------------------
-- optimization loop
--
local losses = {}
local optim_state = {learningRate = 1e-1}

for i = 1, 1000 do
  local _, loss = optim.adagrad(feval, params, optim_state)
  losses[#losses + 1] = loss[1] -- append the new loss

  if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      
  end
end

local outputs, alpha, mu, sigma = unpack(model:forward({features_input, labels_input}))

local alpha_sigma = torch.zeros(alpha:size())
alpha_sigma:cdiv(alpha, sigma)
local y, indexes = torch.max(alpha_sigma,2)

local predictions = labels_input:clone()
for i = 1, predictions:size(1) do 
  predictions[i] = mu[i][{{}, {indexes[i][1]}}]
  
end

print(outputs)
print(alpha)
print(mu)
print(sigma)
print(predictions)



