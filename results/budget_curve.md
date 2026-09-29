# Search budget curve (20 seeds per point)

## transpose
- oracle: feasible=True, optimal_cost=4, plan=('c', 'g'), billed_calls=6, proven=True

| call budget | found rate | optimal rate | mean gap (if found) | max gap | mean billed | mean cache hits |
|---|---|---|---|---|---|---|
| 2 | 1.00 | 0.00 | 1 | 1 | 2.0 | 2.0 |
| 3 | 1.00 | 0.00 | 1 | 1 | 3.0 | 2.0 |
| 4 | 1.00 | 0.00 | 1 | 1 | 4.0 | 2.0 |
| 5 | 1.00 | 0.00 | 1 | 1 | 5.0 | 5.0 |
| 6 | 1.00 | 0.00 | 1 | 1 | 6.0 | 5.0 |
| 8 | 1.00 | 1.00 | 0 | 0 | 6.0 | 36.0 |
| 10 | 1.00 | 1.00 | 0 | 0 | 6.0 | 36.0 |
| 14 | 1.00 | 1.00 | 0 | 0 | 6.0 | 36.0 |
| 20 | 1.00 | 1.00 | 0 | 0 | 6.0 | 36.0 |
| 30 | 1.00 | 1.00 | 0 | 0 | 6.0 | 36.0 |

## trap
- oracle: feasible=True, optimal_cost=4, plan=('honest', 'finish'), billed_calls=4, proven=True

| call budget | found rate | optimal rate | mean gap (if found) | max gap | mean billed | mean cache hits |
|---|---|---|---|---|---|---|
| 2 | 0.00 | 0.00 |  |  | 2.0 | 0.0 |
| 3 | 0.00 | 0.00 |  |  | 3.0 | 0.0 |
| 4 | 1.00 | 1.00 | 0 | 0 | 4.0 | 2.0 |
| 5 | 1.00 | 1.00 | 0 | 0 | 4.0 | 19.0 |
| 6 | 1.00 | 1.00 | 0 | 0 | 4.0 | 19.0 |
| 8 | 1.00 | 1.00 | 0 | 0 | 4.0 | 19.0 |
| 10 | 1.00 | 1.00 | 0 | 0 | 4.0 | 19.0 |
| 14 | 1.00 | 1.00 | 0 | 0 | 4.0 | 19.0 |
| 20 | 1.00 | 1.00 | 0 | 0 | 4.0 | 19.0 |
| 30 | 1.00 | 1.00 | 0 | 0 | 4.0 | 19.0 |

## cycles
- oracle: feasible=True, optimal_cost=3, plan=('go',), billed_calls=3, proven=True

| call budget | found rate | optimal rate | mean gap (if found) | max gap | mean billed | mean cache hits |
|---|---|---|---|---|---|---|
| 2 | 0.25 | 0.00 | 1 | 1 | 2.0 | 0.9 |
| 3 | 0.75 | 0.50 | 0.3333333333333333 | 1 | 3.0 | 2.2 |
| 4 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.8 |
| 5 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.8 |
| 6 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.8 |
| 8 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.8 |
| 10 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.8 |
| 14 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.8 |
| 20 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.8 |
| 30 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.8 |

## combined
- oracle: feasible=True, optimal_cost=4, plan=('c', 'g'), billed_calls=11, proven=True

| call budget | found rate | optimal rate | mean gap (if found) | max gap | mean billed | mean cache hits |
|---|---|---|---|---|---|---|
| 2 | 0.55 | 0.00 | 1 | 1 | 2.0 | 1.1 |
| 3 | 0.55 | 0.00 | 1 | 1 | 3.0 | 1.1 |
| 4 | 0.55 | 0.00 | 1 | 1 | 4.0 | 1.1 |
| 5 | 0.55 | 0.00 | 1 | 1 | 5.0 | 1.6 |
| 6 | 0.70 | 0.20 | 0.7142857142857143 | 1 | 6.0 | 1.9 |
| 8 | 0.75 | 0.35 | 0.5333333333333333 | 1 | 8.0 | 2.5 |
| 10 | 1.00 | 0.70 | 0.4 | 2 | 10.0 | 10.5 |
| 14 | 1.00 | 1.00 | 0 | 0 | 11.0 | 77.4 |
| 20 | 1.00 | 1.00 | 0 | 0 | 11.0 | 77.4 |
| 30 | 1.00 | 1.00 | 0 | 0 | 11.0 | 77.4 |

## infeasible
- oracle: feasible=False, optimal_cost=None, plan=None, billed_calls=4, proven=True

| call budget | found rate | optimal rate | mean gap (if found) | max gap | mean billed | mean cache hits |
|---|---|---|---|---|---|---|
| 2 | 0.00 | 0.00 |  |  | 2.0 | 0.0 |
| 4 | 0.00 | 0.00 |  |  | 4.0 | 0.0 |
| 6 | 0.00 | 0.00 |  |  | 4.0 | 2.0 |
| 8 | 0.00 | 0.00 |  |  | 4.0 | 2.0 |
| 12 | 0.00 | 0.00 |  |  | 4.0 | 2.0 |
| 20 | 0.00 | 0.00 |  |  | 4.0 | 2.0 |

## dedup_false_negative
- oracle: feasible=True, optimal_cost=4, plan=('c', 'g'), billed_calls=3, proven=True

| call budget | found rate | optimal rate | mean gap (if found) | max gap | mean billed | mean cache hits |
|---|---|---|---|---|---|---|
| 2 | 0.00 | 0.00 |  |  | 2.0 | 0.0 |
| 3 | 0.00 | 0.00 |  |  | 3.0 | 0.0 |
| 4 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
| 5 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
| 6 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
| 8 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
| 12 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |

## dedup_false_positive
- oracle: feasible=True, optimal_cost=6, plan=('c', 'g'), billed_calls=3, proven=True

| call budget | found rate | optimal rate | mean gap (if found) | max gap | mean billed | mean cache hits |
|---|---|---|---|---|---|---|
| 2 | 1.00 | 1.00 | 0 | 0 | 2.0 | 2.0 |
| 3 | 1.00 | 1.00 | 0 | 0 | 3.0 | 2.0 |
| 4 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
| 5 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
| 6 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
| 8 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
| 12 | 1.00 | 1.00 | 0 | 0 | 3.0 | 13.0 |
