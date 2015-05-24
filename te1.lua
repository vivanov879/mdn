require 'mobdebug'.start()

require 'nn'
require 'nngraph'

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

m = nn.gModule({features, labels}, {z})
features_input = torch.rand(10, 5)
labels_input = torch.rand(10, 2)
x_input = torch.rand(10, 6)
output = m:forward({features_input, labels_input})
print(output)
print(features_input)
print(labels_input)

--использовать MSECriterion с target=0, потому что там под логарифмом взвешенная с положительными весами сумма нормальных распределний, что больше нуля, то есть отрицат логарифм не будет >= 0

target = torch.zeros(output:size())
criterion = nn.MSECriterion()
loss = 0
loss = loss + criterion:forward(output, target)

print(loss)





a = 1