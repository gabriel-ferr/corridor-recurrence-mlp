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
#           Essa é a recorrência padrão =3
function standard_recurrence(x::AbstractVector{__RP_FLOAT_TYPE}, y::AbstractVector{__RP_FLOAT_TYPE}, ε::Any)
    return (ε - euclidean(x, y)) >= 0 ? 1 : 0
end
# ================================================================================================================================= #
end