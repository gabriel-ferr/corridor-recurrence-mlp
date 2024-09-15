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
# ================================================================================================================================= #
#       - Bibliotecas necessárias para rodar esse script:
#           > Obviamente a RP.jl é uma dependência, sendo ela o arquivo `rp.jl` presente junto com esse =3
#           
# ================================================================================================================================= #
include("rp.jl")
# ================================================================================================================================= #
using .RP
using JLD2
using BSON
using Flux
using Dates
using Statistics
using ProgressMeter
using BenchmarkTools
# ================================================================================================================================= #
#       - Aqui tem algumas constantes para ajudar na configuração facilitada =3

#           Valores de Beta para a classificação.
const β_values = [1.99, 2.39, 2.69, 2.99, 3.39, 3.69, 3.99, 4.39, 4.69, 4.99, 5.39, 5.69, 5.99, 6.39, 6.69, 6.99, 7.39, 7.49, 7.99, 8.39]

#           Tamanho das séries temporais que serão usadas.
const timeseries_size = 1000

#           Épocas de treinamento para a rede neural.
const epochs = 50

#           Tamanho dos motifs (microestados)
const motif_size = 3

#           Número de amostras para as redes neurais.
const mlp_samples = 20

#           Intervalo de threshold que vai ser usado (fica mais fácil deixar isso pronto =3)
#   O terceiro valor disso determina a resolução, mas como o objetivo aqui é calcular a accuracy para todas as entropias
#   deixar uma resolução muito alta vai fazer o computador chorar com a carga de trabalho >.<
const ε = range(0, 0.9995, 100)

#           Tempo de execução mínimo em segundos, aproximadamente....
const run_time = 2

# ================================================================================================================================= #
#           Função de entrada para o programa. É melhor colocar tudo dentro de funções, já que o escopo global do julia
#   pode ser um inferno de lidar...
function main()
    #           Valores para calcular a entropia média =3
    xo_to_entropy = range(0.00001, 0.99999, 5)
    #           Valores para o treinamento da rede.
    xo_to_train_mlp = rand(Float64, 3000)
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
    run_time_prev = DateTime(0) + Second(floor(Int, run_time * ((length(ε) / 2) * (length(ε) - 1) - (status[1] / 2) * (status[1] - 1))))

    println(string("Tempo de execução previsto: ", year(run_time_prev), " anos, ", month(run_time_prev) - 1, " meses, ", day(run_time_prev) - 1, " dias, ", hour(run_time_prev), " horas, ", minute(run_time_prev), " minutos e ", second(run_time_prev), " segundos."))

    # ============================================================================================================================= #
    #
    #           Vamos começar a explorar o computador =D
    #       (e a simetria também ...)
    @showprogress for max_index = status[1]:length(ε)
        for min_index = 1:(max_index-1)
            #       Temos 2 tarefas aqui, a primeira é calcular a entropia e a segunda a accuracy quando aplicado a uma MLP.
            #       Bom.. para tentar agilizar um pouco o processo, vou separar isso em 2 corrotinas, uma para a entropia
            #   e outra para a rede. Eu podia separar ainda mais o processo, ao nível dos threshold, mas acho que fazer a
            #   sincronização disso vai ser um inferno, então, vou-me recusar a fazer, o PC que se resolva =V

            #       Como vou trabalhar com processos assíncronos ...
            @elapsed begin
                task_entropy = @task calculate_entropy(ε[min_index], ε[max_index]; pvec=pow_vector)

                schedule(task_entropy)

                s = fetch(task_entropy)

                etr_obj = load_object("data/beta-x-entropy.dat")
                etr_obj[min_index, max_index, :] .= s
                save_object("data/beta-x-entropy.dat", etr_obj)

                etr_obj = Nothing
            end
        end
        status[1] += 1
        save_object("data/beta-x-status.dat", status)
    end

    println(status)
end
# ================================================================================================================================= #
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