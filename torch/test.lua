require "torch"

local x = torch.rand(2,3)
local y = torch.rand(2)

local data = {data=x, label=y}
print(data.data)
setmetatable(data,
{ __index = function(t, i)
     return {t.data[i], t.label[i]}
end
}
);

function data:size()
   print("--")
   return self.x:size(1)
end
--print(data[1])
print(data.data)
--print(data:size())
