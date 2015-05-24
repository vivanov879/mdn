require 'mobdebug'.start()

require 'nn'
require 'nngraph'

local input_size = 5
local n_mixtures = 3
local x = nn.Identity()()

local h_alpha = nn.Linear(input_size, n_mixtures)(x)
local h_sigma = nn.Linear(input_size, n_mixtures)(x)
local h_mu = nn.Linear(input_size, n_mixtures)(x)

local h_alpha_softmax = nn.SoftMax()(h_alpha)
local h_sigma_exp = nn.Exp()(h_sigma)


local m = nn.gModule({x}, {h_alpha_softmax, h_mu, h_sigma_exp})
graph.dot(m.fg, 'MDN')



