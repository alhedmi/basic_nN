function plot_dynamic_network2(ax, arch, w)


    cla(ax,'reset');
    hold(ax,'on');

    % 1) Compute the (x,y) positions of each layer
    positions = calculate_positions(arch); 

    % 2) Break w into blocks dynamically
    wBlock = cell(1, length(arch)-1);
    bBlock = cell(1,length(arch));
    bBlock{1} = [1;1];
    startIdx = 1;
    for i = 1:length(arch)-1
        wNr = arch(i) * arch(i+1);
        wBlock{i} = w(startIdx:startIdx + wNr - 1);
        bBlock{i+1} = w(startIdx + wNr - 1  :  startIdx + wNr + arch(i+1) -1);
        startIdx = startIdx + wNr + arch(i+1);
    end
    disp(bBlock{1})
    % 3) Draw lines for each layer-to-layer
    for iLayer = 1 : (length(arch)-1)
        block = wBlock{iLayer};
        layerFrom = positions{iLayer};
        layerTo   = positions{iLayer+1};

        % Iterate over each connection dynamically
        idx = 1;
        for fromNeuron = 1:arch(iLayer)
            for toNeuron = 1:arch(iLayer+1)
                wVal = block(idx);
                draw_connection(ax, layerFrom(fromNeuron, 1), layerFrom(fromNeuron, 2),...
                                    layerTo(toNeuron, 1), layerTo(toNeuron, 2), wVal);
                idx = idx + 1;
            end
        end
    end

    % 4) Draw neurons with different colors for each layer. 
    colors = {'k', 'g', 'c', 'm', 'c', 'm'}; % Extend for more layers
    for iLayer = 1:length(arch)
        for i = 1:arch(iLayer)

            scatter(ax, positions{iLayer}(i,1), positions{iLayer}(i,2), ...
                  500  , colors{mod(iLayer-1, length(colors))+1}, 'filled'); % dynamic size : abs(bBlock{iLayer}(i)* 800)
        end
    end

    % 5) Final formatting
    title(ax, 'Network Weights During Training');
    xlabel(ax, 'Layer');
    ylabel(ax, 'Neuron Position');
    xlim(ax, [0, (length(arch)-1)*1]);
    axis(ax, 'equal');
    grid(ax, 'on');
    hold(ax,'off');
end


%---------------------------------------------------
%  SUBFUNCTION:  draw_connection
%---------------------------------------------------
function draw_connection(ax, x1, y1, x2, y2, wVal)
    if wVal > 0
        color = [0 0 1];  % blue if positive
    else
        color = [1 0 0];  % red if negative
    end
    lineWidth = max(abs(wVal)*2, 1);%0.5
    plot(ax, [x1 x2], [y1 y2], 'Color', color, 'LineWidth', lineWidth);
end


%---------------------------------------------------
%  SUBFUNCTION:  calculate_positions
%---------------------------------------------------
function positions = calculate_positions(layers)
    % layers = [2, 4, 3]
    
    xSpacing = 3; 
    ySpacing = 1; 
    positions = cell(1, length(layers));

    for i = 1:length(layers)
        x = (i-1) * xSpacing;
        y = linspace( ...
             -ySpacing * (layers(i)-1)/2, ...
              ySpacing * (layers(i)-1)/2, ...
              layers(i));
        positions{i} = [repmat(x, layers(i), 1), y'];
    end
end
