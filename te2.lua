require 'mobdebug'.start()

require 'nn'
require 'nngraph'
require 'optim'

local n_gaussians = 2
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
  sigma_sq_inv = nn.Power(-2)(sigma_select)
  alpha_select = nn.Select(2,i)(alpha)
  mm = nn.CMulTable()({s, sigma_sq_inv})
  e = nn.Exp()(mm)
  sigma_mm = nn.Power(-n_labels)(sigma_select)
  r = nn.CMulTable()({e, sigma_mm, alpha_select})
  r = nn.MulConstant(math.pow((2 * math.pi), -0.5*n_labels))(r)
  l[#l + 1] = r
end

z3 = nn.CAddTable()(l)
z4 = nn.Log()(z3)
z5 = nn.MulConstant(-1)(z4)

m = nn.gModule({features, labels}, {z5, alpha, mu, sigma, e, l[1], l[2]})



criterion = nn.MSECriterion()


--заделаем тестовые features и labels
local n_data = 100
local features_input = torch.zeros(n_data, n_features)
local labels_input = torch.zeros(n_data, n_labels)
for i = 1, n_data do 
  element = torch.ones(n_features)

  element:mul(i)
  element[2] = 0.5
  element[3] = 1
  element:add(torch.randn(element:size()))
  element:div(n_data)
  features_input[{{i}, {}}] = element
  
  label = torch.ones(n_labels)
  label:mul( i)
  label[1] = 0.1
  label:add(torch.randn(label:size()))
  label:div(n_data)
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
  local output, alpha, mu, sigma, e, l1, l2 = unpack(model:forward({features_input, labels_input}))
  print(torch.any(output:lt(0)))
  local loss = torch.mean(output)
  -- backward
  local dsigma = torch.zeros(sigma:size())
  local dalpha = torch.zeros(alpha:size())
  local dmu = torch.zeros(mu:size())
  local doutput = torch.ones(output:size())
  model:backward({features_input, labels_input}, {doutput, dalpha, dmu, dsigma, torch.zeros(output:size()), torch.zeros(output:size()), torch.zeros(output:size())})

  return loss, grads
end

------------------------------------------------------------------------
-- optimization loop
--
local losses = {}
local optim_state = {learningRate = 1e-1}

for i = 1, 10000 do
  local _, loss = optim.adagrad(feval, params, optim_state)
  losses[#losses + 1] = loss[1] -- append the new loss

  if i % 10 == 0 then
      print(string.format("iteration %4d, loss = %6.6f", i, loss[1]))
      --print(params)
      
  end
end

-- вместо torch.zeros(labels_input:size()) можно ставить любую миатрицу нужных размеров, потому что она не влияет на нужные нам alpha, mu, sigma
local outputs, alpha, mu, sigma = unpack(model:forward({features_input, torch.zeros(labels_input:size())}))

local alpha_sigma = torch.zeros(alpha:size())
alpha_sigma:cdiv(alpha, sigma)
local y, indexes = torch.max(alpha_sigma,2)

local predictions = labels_input:clone()
for i = 1, predictions:size(1) do 
  predictions[i] = mu[i][{{}, {indexes[i][1]}}]
  
end

print((alpha)[{{60}, {}}])
print((mu)[{{60}, {}}])
print((sigma)[{{60}, {}}])
print(labels_input[{{20}, {}}])
print(predictions[{{20}, {}}])



