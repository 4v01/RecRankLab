"""Validation-only epoch selection; final refit is performed by the caller."""
import math


class EarlyStopping:
    def __init__(self, evaluate, metric, interval=5, patience=3, min_delta=0.):
        if metric not in ('rmse','ndcg@10') or interval<1 or patience<1 or min_delta<0:
            raise ValueError('Invalid stopping configuration')
        self.evaluate=evaluate;self.metric=metric;self.interval=interval
        self.patience=patience;self.min_delta=min_delta
        self.best=None;self.bad_checks=0;self.history=[]

    def __call__(self,model,epoch):
        if epoch%self.interval and epoch!=model.config['epochs']:return False
        metrics=self.evaluate(model)
        value=metrics[self.metric]
        if value is None or not math.isfinite(value):raise ValueError('Nonfinite validation objective')
        record={'epoch':epoch,'metrics':metrics}
        self.history.append(record)
        better=self.best is None
        if self.best is not None:
            previous=self.best['metrics'][self.metric]
            better=value<previous-self.min_delta if self.metric=='rmse' else value>previous+self.min_delta
        if better:self.best=record;self.bad_checks=0
        else:self.bad_checks+=1
        print(f"  epoch={epoch} validation_{self.metric}={value:.6f} best_epoch={self.best['epoch']}",flush=True)
        return self.bad_checks>=self.patience
