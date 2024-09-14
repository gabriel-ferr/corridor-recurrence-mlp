#
#               [Portuguese Version]
#           Por Gabriel Ferreira
#               Orientação: Thiago de Lima Prado
#                           Sérgio Roberto Lopes
# ================================================================================================================================= #
#
#           Esse script é basicamente uma biblioteca que eu montei para trabalhar com RP no Julia.
#           Ela não está no Pkg repository (até porque eu não compilei ela, e nem sei fazer isso, depois eu descubro =V)
#   então, fazer a importação desse script é um pouco diferente. Para fazer isso é só fazer algo como:
# 
#           julia> include("rp.jl")
#           julia> using .RP
#
#           Eu deixei essa biblioteca bem enxuta, focada apenas na parte de cálculo, então, para desenvolver
#   gráficos é necessário usar algo como o CairoMakie ou Plots. Também não criei estruturas complexas e abstratas,
#   como o Recurrence Analysis usado pelo System Dynamics faz, tentei deixar bem simples =3
#
#           Vou tentar optimizar o código e construir algumas operações mult-thread para agilizar o processo ... vamos
#   ver no que dá >.<
# ================================================================================================================================= #
#       - Versão 1.0
#           * Calcula a matriz de recorrência a partir de uma série temporal, ou seja, um RP.
#           * Calcula a matriz de recorrência a partir de duas séries temporais, ou seja, um CRP.
#
#       - Versão 1.1
#           * Calcula os microestados de recorrência de um RP.
#               > A estrutura disso é meio chatinha de fazer uma certa abstração, porque eu acabei colocando tudo em
#   uma única função, então, ficou um pouquinho carregada, mas deve funcionar ... eu acho =V
#           * Calcula a Entropia de Shenon a partir desses microestados.
# ================================================================================================================================= #
#       - Bibliotecas necessárias para rodar esse script:
#           > Distances.jl
#           > LinearAlgebra.jl
# ================================================================================================================================= #
module RP
# ================================================================================================================================= #
#           Deixei por padrão o uso de Float64, então, se quiser usar outra precisão altere abaixo:
const __RP_FLOAT_TYPE = Float64
# ================================================================================================================================= #
using Distances
using StatsBase
using LinearAlgebra
# ================================================================================================================================= #
export recurrence_matrix
#
#           Essa função é bem simples, calcula a nossa matriz de recorrência, retornando ela.
#           A série de entrada deve ser uma array de 2 dimensões, onde cada coluna representa um vetor posição para um instante
#   t da série temporal, e cada linha representa a coordenada para cada dimensão do espaço. Ou seja, segue o padrão de definição
#   de um vetor como uma matriz coluna.
function recurrence_matrix(serie::AbstractArray{__RP_FLOAT_TYPE,2}, ε::Any; recurrence=standard_recurrence)
    sz = size(serie)
    rp = zeros(Int8, (sz[2], sz[2]))        # O número de colunas é basicamente o tempo da série =3

    for j = 1:sz[2]
        for i = 1:j
            rp[i, j] = recurrence(serie[:, i], serie[:, j], ε)
        end
    end

    return Symmetric(rp)
end
# ================================================================================================================================= #
export crossrecurrence_matrix
#
#           Função para cross recurrence. Apesar da recorrência de uma única série ser um caso particular
#   da CRP, optei por criar duas funções distintas, já que dá pra economizar processamento no caso do RP, que é simétrico >.<
function crossrecurrence_matrix(serie_one::AbstractArray{__RP_FLOAT_TYPE,2}, serie_two::AbstractArray{__RP_FLOAT_TYPE,2}, ε::Any; recurrence=standard_recurrence)
    sz_one = size(serie_one)
    sz_two = size(serie_two)
    rp = zeros(Int8, (sz_one[2], sz_two[2]))            # Mesma lógica, mas o tamanho pode ser distinto, já que as séries são diferentes.

    for i = sz_one[2]
        for j = sz_two[2]
            rp[i, j] = recurrence(serie_one[:, i], serie_two[:, j], ε)
        end
    end

    return rp
end
# ================================================================================================================================= #
export motifs_probabilities
#
#           Bom, pensando em uma expansão futura, nomeei essa função com o conceito "probabilidades dos motifs", ou algo do gênero.
#           Basicamente, isso pega o rp e calcula as probabilidades dos microestados como estamos normalmente acostumados.
#
#   Importante! Configurar essa função pode ser um pouco chatinho, então, segue uma descrição melhor dos parametros:
#
#           1. no_use_samples (true/false) - se verdadeiro ele vai pegar TODOS os motifs existentes no RP para calcular
#   as probabilidades, então, pode demorar mais, só que vai resultar em uma maior precisão; se falso, ele vai
#   pegar essas probabilidades partindo de amostras aleatórias dentro de uma quantidade definida.
#
#           2. samples (Tuple{Int, Int}) - no caso de usar amostragem, pega uma quantidade fixa de valores de cada linha e coluna
#   definidos nos 2 valores passados nessa Tuple, sendo [1] para linhas e [2] para colunas. Por padrão isso está em 20% do tamanho do
#   RP.
#
#           3. power_vector (Vector{Int}) - recomendo que seja instânciado com antecedência se você for colocar isso em um for,
#   vai poupar uma bela quantidade de tempo com o GC liberando e realocando memória para a mesma coisa... e sério, esse negócio
#   pode ocupar um bom tempo fazendo isso =.=
function motifs_probabilities(rp::AbstractMatrix{Int8}, n::Int; no_use_samples::Bool=false, samples::Tuple{Int,Int}=(floor(Int, size(rp, 1) * 0.2), floor(Int, size(rp, 2) * 0.2)), power_vector=power_vector(n))
    motifs = Dict{Int64,__RP_FLOAT_TYPE}()      # Bom, como existe uma quantidade de estados não usados para a maioria dos sistemas quando partimos de n >= 3, eu usei um Dict aqui, para economizar memória =3
    #                                           # Com isso aqui dá pra calcular as probabilidades de, sei lá, n = 10, mas obviamente usar isso pode ser um pouco incoveniente hehe

    p = 0
    add = 0
    a_bin = 0
    counter = 0
    sz = size(rp)

    #       Bom, aqui vem o primeiro BO, juntar o conceito de amostragem com o fato de que eu posso querer pegar
    #   todos os microestados =V
    index_x = missing
    index_y = missing

    if (no_use_samples)
        index_x = 1:(sz[1]-(n-1))
        index_y = 1:(sz[2]-(n-1))
    else
        index_x = sample(1:sz[1]-(n-1), samples[1])
    end

    for i in index_x
        if (!no_use_samples)
            index_y = sample(1:sz[2]-(n-1), samples[2])
        end

        for j in index_y
            add = 0

            for x = 1:n
                for y = 1:n
                    a_bin = rp[i+x-1, j+y-1]
                    add = add + a_bin * power_vector[y+(n*(x-1))]
                end
            end

            p = Int64(add) + 1
            motifs[p] = get(motifs, p, 0) + 1
            counter += 1
        end
    end

    for k in keys(motifs)
        motifs[k] /= counter
    end

    return motifs, counter
end
# ================================================================================================================================= #
export entropy
#
#           Calcula a entropia de Shenon para as recorrências. (ou melhor, aplica isso na entropia de Shenon)
function entropy(probs::Dict{Int64,__RP_FLOAT_TYPE})
    s = 0.0

    for k in keys(probs)
        if (probs[k] > 0)
            s += (-1) * probs[k] * log(probs[k])
        end
    end

    return s
end
# ================================================================================================================================= #
export power_vector
#
#           Isso aqui serve como uma base para converser binário para decimal =V
function power_vector(n::Int)
    vec = zeros(Int64, (n * n))

    for i in eachindex(vec)
        vec[i] = Int64(2^(i - 1))
    end

    return vec
end
# ================================================================================================================================= #
#           Essa é a recorrência padrão =3
function standard_recurrence(x::AbstractVector{__RP_FLOAT_TYPE}, y::AbstractVector{__RP_FLOAT_TYPE}, ε::__RP_FLOAT_TYPE)
    return (ε - euclidean(x, y)) >= 0 ? 1 : 0
end
# ================================================================================================================================= #
#           Recorrência de corredor =3
function corridor_recurrence(x::AbstractVector{__RP_FLOAT_TYPE}, y::AbstractVector{__RP_FLOAT_TYPE}, ε::Tuple{__RP_FLOAT_TYPE,__RP_FLOAT_TYPE})
    return ((ε[2] - euclidean(x, y)) >= 0 ? 1 : 0) * ((euclidean(x, y) - ε[1]) >= 0 ? 1 : 0)
end
# ================================================================================================================================= #
end