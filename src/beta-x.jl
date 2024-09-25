#
#               [Portuguese Version]
#           Por Gabriel Ferreira
#               Orientação: Thiago de Lima Prado
#                           Sérgio Roberto Lopes
# ================================================================================================================================= #
#
#           Aqui temos o código para o Beta-X (ou Bernoulli Shift Generealized), aplicando isso a uma MLP para comparar
#   a recorrência padrão com a recorrência de corredor =3
#
#           Vou fazer a seguinte sequência aqui:
#
#       1. PREPARAÇÃO
#           a) Pega um conjunto de valores iniciais (x0) entre 0.00001 e 0.99999, usados para calcular a entropia.
#   Imagino que 5 amostras nesse caso é suficiente.
#           b) Pega um outro conjunto, agora aleatório, de valores iniciais, usados para o treinamento da rede neural.
#           c) Por fim, separa um conjunto de valores para testar a rede, sendo que nenhum dos valores nesse conjunto
#   está contido no conjunto de treinamento.
#           d) Pega esses dados e gera as séries temporais do Beta-X =V
#           e) Usa um pouco de gambiarra para carregar e salvar os dados gerados, assim como as matrizes de saída hehe
#
#       2. CORRIDOR RECURRENCE (Ou CRD Recurrence, para abreviar =V)
#           A ideia aqui é abusar do GC do Julia, já que esse é o único jeito de deixar memória livre, então,
#   talvez acabe aumentando um pouquinho o tempo de processamento, mas vamos ficar transitando informação do HD para a RAM
#   e realocando memória para evitar problemas =V (até pq pelas minhas contas a matriz de accuracy vai consumir 1GB do jeito que fiz ela <,<)
#
#           -- As descrições disso estão ali pela linha 150 +/-
#
#       OBS: Conforme eu fui escrevendo o código os comentários ficaram mais avulsos, estou a disposição para quem tiver dúvidas =3
#
# ================================================================================================================================= #
#       - Bibliotecas necessárias para rodar esse script:
#           > Obviamente a RP.jl é uma dependência, sendo ela o arquivo `rp.jl` presente junto com esse =3
#           
# ================================================================================================================================= #
include("rp.jl")
# ================================================================================================================================= #
using .RP
using JLD2
using Flux
using Statistics
using LinearAlgebra
using BenchmarkTools
# ================================================================================================================================= #
#       - Aqui tem algumas constantes para ajudar na configuração facilitada =3

#           Valores de Beta para a classificação.
const β_values = [2.59, 2.99, 3.59, 3.99, 4.59, 4.99, 5.59, 5.99, 6.59, 6.99]

#           Tamanho das séries temporais que serão usadas.
const timeseries_size = 500

#           Épocas de treinamento para a rede neural.
const epochs = 50

#           Tamanho dos motifs (microestados)
const motif_size = 2

#           Número de amostras para as redes neurais.
const mlp_samples = 1

#           Intervalo de threshold que vai ser usado (fica mais fácil deixar isso pronto =3)
#   O terceiro valor disso determina a resolução, mas como o objetivo aqui é calcular a accuracy para todas as entropias
#   deixar uma resolução muito alta vai fazer o computador chorar com a carga de trabalho >.<
const ε = range(0, 0.9995, 25)

#           Tempo de execução mínimo em segundos, aproximadamente....
#const run_time = 520.36

# ================================================================================================================================= #
#           Função de entrada para o programa. É melhor colocar tudo dentro de funções, já que o escopo global do julia
#   pode ser um inferno de lidar...
function main()
    #           Valores para calcular a entropia média =3
    xo_to_entropy = range(0.00001, 0.99999, 5)
    #           Valores para o treinamento da rede.
    xo_to_train_mlp = rand(Float64, 750)
    #           Valores para o teste da rede.
    xo_to_test_mlp = rand(Float64, floor(Int, length(xo_to_train_mlp) / 3))

    #           Vou garantir que a parte 1.c) seja completa aqui, trocando os valores de teste que estejam contidos nos de treinamento.
    for i in eachindex(xo_to_test_mlp)
        while (xo_to_test_mlp[i] in xo_to_train_mlp)
            new_value = rand(Float64, 1)
            while (new_value in xo_to_test_mlp)
                new_value = rand(Float64, 1)
            end
            xo_to_test_mlp[i] = new_value
        end
    end

    #       Bom, agora vem a questão, já que esse negócio pode demorar MUITO, vou verificar se já teve algum
    #   processo feito previamente ... sim, eu sei, eu deveria fazer isso primeiro, mas fazer oq, to tentando
    #   deixar o código simples =/
    #
    #       OBS: Essa gambiarra tá custando +/- 0.001018 segundo =3 (ou +/- 9.316374 segundos, se ele for gerar os dados hehe)
    if (!isfile("data/beta-x-entropy-serie.dat"))
        serie_to_entropy = β(xo_to_entropy)
        save_object("data/beta-x-entropy-serie.dat", serie_to_entropy)
        xo_to_entropy = Nothing
        serie_to_entropy = Nothing
    end
    if (!isfile("data/beta-x-mlp-test-serie.dat"))
        serie_to_test_mlp = β(xo_to_test_mlp)
        save_object("data/beta-x-mlp-test-serie.dat", serie_to_test_mlp)
        xo_to_test_mlp = Nothing
        serie_to_test_mlp = Nothing
    end
    if (!isfile("data/beta-x-mlp-train-serie.dat"))
        serie_to_train_mlp = β(xo_to_train_mlp)
        save_object("data/beta-x-mlp-train-serie.dat", serie_to_train_mlp)
        xo_to_train_mlp = Nothing
        serie_to_train_mlp = Nothing
    end

    #       Bom, vamos criar um arquivo para armazenar as entropias também =3
    if (!isfile("data/beta-x-entropy.dat"))
        entropy_data = zeros(Float64, length(ε), length(ε), length(β_values))
        save_object("data/beta-x-entropy.dat", entropy_data)
        entropy_data = Nothing
    end

    #       E aqui a matriz de accuracy que vai consumir um rim em espaço de armazenamento ... desculpa computador >.<
    if (!isfile("data/beta-x-accuracy.dat"))
        accuracy_data = zeros(Float64, length(ε), length(ε), epochs, mlp_samples)
        save_object("data/beta-x-accuracy.dat", accuracy_data)
        accuracy_data = Nothing

        #   Gera as redes neurais =3
        for index = 1:mlp_samples
            model = Chain(
                Dense(2^(motif_size * motif_size) => 128, identity),
                Dense(128 => 64, selu),
                Dense(64 => 32, selu),
                Dense(32 => length(β_values)),
                softmax
            )

            save_object(string("network/beta-x-", index, ".mlp"), f64(model))
        end
    end

    #      Vamos criar um arquivo para salvar o progresso também...
    if (!isfile("data/beta-x-status.dat"))
        #       O primeiro valor registra os indices do threshold, e o segundo o indice do Beta =D
        save_object("data/beta-x-status.dat", [1, 1])
    end
    # ============================================================================================================================= #
    #       Vou instanciar o vetor de potência aqui para economizar memória.
    pow_vector = power_vector(motif_size)

    #       Previsão do tempo de execução
    status = load_object("data/beta-x-status.dat")
    #run_time_prev = DateTime(0) + Second(floor(Int, run_time * ((length(ε) / 2) * (length(ε) - 1) - (status[1] / 2) * (status[1] - 1))))

    #println(string("Tempo de execução previsto: ", year(run_time_prev), " anos, ", month(run_time_prev) - 1, " meses, ", day(run_time_prev) - 1, " dias, ", hour(run_time_prev), " horas, ", minute(run_time_prev), " minutos e ", second(run_time_prev), " segundos."))

    # ============================================================================================================================= #
    #
    #           Vamos começar a explorar o computador =D
    #       (e a simetria também ...)
    for max_index = status[1]:length(ε)
        for min_index = status[2]:(max_index-1)
            #
            #       - Primeiro calcula a entropia, que demora menos =3
            etrpy = calculate_entropy(ε[min_index], ε[max_index]; pvec=pow_vector)
            etr_obj = load_object("data/beta-x-entropy.dat")
            etr_obj[min_index, max_index, :] .= etrpy
            save_object("data/beta-x-entropy.dat", etr_obj)

            etrpy = Nothing
            etr_obj = Nothing

            #
            #       - Vamos para a accuracy agora...
            accr = calculate_accuracy(ε[min_index], ε[max_index]; pvec=pow_vector)
            acr_obj = load_object("data/beta-x-accuracy.dat")
            acr_obj[min_index, max_index, :, :] .= accr
            save_object("data/beta-x-accuracy.dat", acr_obj)

            accr = Nothing
            acr_obj = Nothing

            status[2] += 1
            save_object("data/beta-x-status.dat", status)
        end
        status[1] += 1
        status[2] = 1
        save_object("data/beta-x-status.dat", status)
    end
    # ============================================================================================================================= #
    #make_graphs()
end
# ================================================================================================================================= #
#function make_graphs()
    #       Gráfico da accuracy
#    accr_mat = load_object("data/beta-x-accuracy.dat")
#    println(findmax(accr_mat[:, :, end, end]'))
#    heatmap(ε, ε, accr_mat[:, :, end, end]', colormap=:ice)
#end
# ================================================================================================================================= #
#           Calcula a accuracy =3
function calculate_accuracy(ε_min, ε_max; pvec=power_vector(motif_size))

    #       Carrega os dados.
    test_serie = load_object("data/beta-x-mlp-test-serie.dat")
    train_serie = load_object("data/beta-x-mlp-train-serie.dat")
    test_sz = size(test_serie, 3)
    train_sz = size(train_serie, 3)

    #       Aloca espaço para salvar as probabilidades.
    test_probs = zeros(Float64, 2^(motif_size * motif_size), test_sz * length(β_values))
    train_probs = zeros(Float64, 2^(motif_size * motif_size), train_sz * length(β_values))

    #       Agora para as labels...
    test_labels = ones(Float64, test_sz * length(β_values))
    train_labels = ones(Float64, train_sz * length(β_values))

    #
    #       Vou criar uma função semelhante a __async_calculate_entropy que usei para o calculate_entropy,
    #   só que essa aqui retorna as probabilidades... o que fica um pouquinho mais chato já que tem
    #   que redimensionar o Dict para uma Array ... apesar de não ser tão complicado fazer isso...
    function __async_calculate_probs(data, ε_tuple; pvec=power_vector(motif_size))
        sz = size(data, 3)
        async_result = zeros(Float64, 2^(motif_size * motif_size), sz)
        for i = 1:sz
            rp = recurrence_matrix(data[:, :, i], ε_tuple; recurrence=RP.corridor_recurrence)
            probs, _ = motifs_probabilities(rp, motif_size; no_use_samples=false, power_vector=pvec)
            async_result[collect(keys(probs)), i] .= collect(values(probs))
        end

        return async_result
    end

    #
    #       Vou criar uma função para efetivamente calcular a accuracy a partir da rede =3
    function __calc_accurarcy(predicted, trusty)
        #       Gera uma matriz de confusão...
        conf = zeros(Int, length(β_values), length(β_values))
        sz = size(predicted, 2)

        for i = 1:sz
            mx_prd = findmax(predicted[:, i])
            mx_trt = findmax(trusty[:, i])
            conf[mx_prd[2], mx_trt[2]] += 1
        end

        return tr(conf) / sum(conf)
    end

    #
    #       Treinamento assíncrono =v
    function __async_net_train(mlp_index, data_loader, test_probs, test_labels)
        accr = zeros(Float64, epochs)
        model = load_object(string("network/beta-x-", mlp_index, ".mlp"))
        model_state = Flux.setup(Flux.Adam(0.0001), model)

        for epc = 1:epochs
            for (x, y) in data_loader
                _, grads = Flux.withgradient(model) do m
                    y_hat = m(x)
                    Flux.logitcrossentropy(y_hat, y)
                end
                Flux.update!(model_state, model, grads[1])
            end

            #       Calcula o accuracy...
            accr[epc] = __calc_accurarcy(model(test_probs), test_labels)
        end

        return accr
    end

    for beta in eachindex(β_values)
        #
        #       Separa as tarefas em blocos para as probabilidades de treinamento...
        train_chuncks = Iterators.partition(1:train_sz, train_sz ÷ Threads.nthreads())
        train_tasks = map(train_chuncks) do chunck
            Threads.@spawn __async_calculate_probs(train_serie[:, :, chunck, beta], (ε_min, ε_max); pvec=power_vector(motif_size))
        end

        train_segment = fetch.(train_tasks)

        test_chunck = Iterators.partition(1:test_sz, test_sz ÷ Threads.nthreads())
        test_tasks = map(test_chunck) do chunck
            Threads.@spawn __async_calculate_probs(test_serie[:, :, chunck, beta], (ε_min, ε_max); pvec=power_vector(motif_size))
        end

        test_segment = fetch.(test_tasks)

        __temp_train = zeros(Float64, 2^(motif_size * motif_size), train_sz)
        __temp_test = zeros(Float64, 2^(motif_size * motif_size), test_sz)

        for i = 1:Threads.nthreads()-1
            __temp_train[:, 1+(i-1)*size(train_segment[i], 2):i*size(train_segment[i], 2)] .= train_segment[i]
            __temp_test[:, 1+(i-1)*size(test_segment[i], 2):i*size(test_segment[i], 2)] .= test_segment[i]
        end

        __temp_train[:, end-(size(train_segment[Threads.nthreads()], 2)-1):end] .= train_segment[Threads.nthreads()]
        __temp_test[:, end-(size(test_segment[Threads.nthreads()], 2)-1):end] .= test_segment[Threads.nthreads()]

        train_probs[:, 1+(beta-1)*train_sz:beta*train_sz] .= __temp_train
        test_probs[:, 1+(beta-1)*test_sz:beta*test_sz] .= __temp_test

        train_labels[1+(beta-1)*train_sz:beta*train_sz] *= β_values[beta]
        test_labels[1+(beta-1)*test_sz:beta*test_sz] *= β_values[beta]
    end

    #       Partiu liberar a memória...
    test_serie = Nothing
    train_serie = Nothing

    #       Prepara para as redes neurais =3
    test_labels = Flux.onehotbatch(test_labels, β_values)
    train_labels = Flux.onehotbatch(train_labels, β_values)

    data_loader = Flux.DataLoader((train_probs, train_labels), batchsize=50, shuffle=true) #|> gpu
    #test_probs = test_probs |> gpu

    #       Aloca espaço para registrar a accuracy...
    accuracy = zeros(Float64, epochs, mlp_samples)

    #       Carrega as redes neurais e treina...
    #train_tasks = map(1:mlp_samples) do chunck
    #    Threads.@spawn __async_net_train(chunck, data_loader, test_probs, test_labels)
    #end

    #mlp_accr = fetch.(train_tasks)

    #       !!! Eu vou supor que o número de redes a serem treinadas é MENOR OU IGUAL ao número de Threads disponíveis !!!
    #       Se isso não for válido, tem que arrumar o código =V
    #       Fiz isso pq é mais fácil =P
    for mlp in 1:mlp_samples
        accuracy[:, mlp] .= __async_net_train(mlp, data_loader, test_probs, test_labels)
    end

    #       Retorna a accuracy
    return accuracy
end
# ================================================================================================================================= #
#           Calcula a entropia =3
function calculate_entropy(ε_min, ε_max; pvec=power_vector(motif_size))

    #   Carrega os dados para o cálculo da entropia.
    data = load_object("data/beta-x-entropy-serie.dat")
    sz = size(data)

    #   Resultados
    result = zeros(Float64, sz[3], sz[4])

    #   Vou criar aqui uma função interna, assíncrona, que pega uma série temporal e calcula a entropia =V
    #   Um pouquinho de gambiarra na tentativa de optimizar isso -.-
    function __async_calculate_entropy(data, ε_tuple; pvec=power_vector(motif_size))
        async_return = zeros(Float64, size(data, 3))

        for beta in eachindex(async_return)
            rp = recurrence_matrix(data[:, :, beta], ε_tuple; recurrence=RP.corridor_recurrence)
            probs, _ = motifs_probabilities(rp, motif_size; no_use_samples=true, power_vector=pvec)
            async_return[beta] = entropy(probs)
        end

        return async_return
    end

    for index = 1:sz[3]
        #       Separa em chuncks
        chuncks = Iterators.partition(1:sz[4], sz[4] ÷ Threads.nthreads())
        tasks = map(chuncks) do chunck
            Threads.@spawn __async_calculate_entropy(data[:, :, index, chunck], (ε_min, ε_max); pvec=pvec)
        end
        etr = fetch.(tasks)
        result[index, 1:length(etr[1])] = etr[1]
        for beta = 2:Threads.nthreads()
            prev_length = 0
            for x = 1:(beta-1)
                prev_length += length(etr[x])
            end

            result[index, 1+prev_length:prev_length+length(etr[beta])] = etr[beta]
        end
    end

    return_vector = zeros(Float64, sz[4])

    for index in eachindex(return_vector)
        return_vector[index] = mean(result[:, index])
    end

    return return_vector
end
# ================================================================================================================================= #
#               Função Bernoulli Shift Generealized, ou Beta-X para os intimos =V
function β(x; transient=round(Int, (10 * timeseries_size)))
    serie = zeros(Float64, (1, timeseries_size, length(x), length(β_values)))

    for β_index in eachindex(β_values)
        for index in eachindex(x)
            before = x[index]

            for time = 1:(timeseries_size+transient)
                after = before * β_values[β_index]

                while (after > 1.0)
                    after = after - 1.0
                end

                before = after

                if (time > transient)
                    serie[1, time-transient, index, β_index] = before
                end
            end

        end
    end

    return serie
end
# ================================================================================================================================= #
@time main()