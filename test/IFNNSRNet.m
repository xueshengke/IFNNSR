function [ output_data ] = IFNNSRNet( model, weights, input_data )

[hei, wid, channel] = size(input_data);

%% update data dimensions in the first lines of the prototxt
fidin = fopen(model, 'r+');
i = 0;
while ~feof(fidin)
    tline = fgetl(fidin);
    i = i + 1;
    newtline{i} = tline;
    if i == 9
        newtline{i} = [tline(1:11) num2str(channel)];
    end
    if i == 10
        newtline{i} = [tline(1:11) num2str(wid)];
    end
    if i == 11
        newtline{i} = [tline(1:11) num2str(hei)];
    end
end
fclose(fidin);

%% write the new dimension in to the prototxt
fidin = fopen(model, 'w+');
for j = 1 : i
    fprintf(fidin, '%s\n', newtline{j});
end
fclose(fidin);

%% create net and load weights
net = caffe.Net(model, weights, 'test'); 

%% feedforward computation
result = net.forward({input_data});
output_data = result{1};
caffe.reset_all();

end
% eltwise_9 = net.layers('ElementWiseProduct9').params(1).get_data();
% add_layer = net.layers('Convolution9').params(1).get_data();
