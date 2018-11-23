import torch
import torch.nn as nn
from utils import to_var, pad, normal_kl_div, normal_logpdf, bag_of_words_loss, to_bow, EOS_ID
import layers
import numpy as np
import random
import pdb

VariationalModels = ['VHRED', 'VHCR','VHCR_new']

class HRED(nn.Module):
    def __init__(self, config):
        super(HRED, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         config.rnn,
                                         config.num_layers,
                                         config.bidirectional,
                                         config.dropout)

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)
        self.context_encoder = layers.ContextRNN(context_input_size,
                                                 config.context_size,
                                                 config.rnn,
                                                 config.num_layers,
                                                 config.dropout)

        self.decoder = layers.DecoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.decoder_hidden_size,
                                         config.rnncell,
                                         config.num_layers,
                                         config.dropout,
                                         config.word_drop,
                                         config.max_unroll,
                                         config.sample,
                                         config.temperature,
                                         config.beam_size)

        self.context2decoder = layers.FeedForward(config.context_size,
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1,
                                                  activation=config.activation)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def forward(self, input_sentences, input_sentence_length,
                input_conversation_length, target_sentences, decode=False):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.size(0)
        max_len = input_conversation_length.data.max().item()

        # encoder_outputs: [num_sentences, max_source_length, hidden_size * direction]
        # encoder_hidden: [num_layers * direction, num_sentences, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(input_sentences,
                                                       input_sentence_length)

        # encoder_hidden: [num_sentences, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(num_sentences, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1])), 0)

        # encoder_hidden: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l), max_len)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        # context_outputs: [batch_size, max_len, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden,
                                                                    input_conversation_length)

        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])

        # project context_outputs to decoder init state
        decoder_init = self.context2decoder(context_outputs)

        # [num_layers, batch_size, hidden_size]
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:

            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)
            return decoder_outputs

        else:
            # decoder_outputs = self.decoder(target_sentences,
            #                                init_h=decoder_init,
            #                                decode=decode)
            # return decoder_outputs.unsqueeze(1)
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)

            # Get top prediction only
            # [batch_size, max_unroll]
            # prediction = prediction[:, 0]

            # [batch_size, beam_size, max_unroll]
            return prediction

    def generate(self, context, sentence_length, n_context):
        # context: [batch_size, n_context, seq_len]
        batch_size = context.size(0)
        # n_context = context.size(1)
        samples = []

        # Run for context
        context_hidden=None
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :],
                                                           sentence_length[:, i])

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            # context_outputs: [batch_size, 1, context_hidden_size * direction]
            # context_hidden: [num_layers * direction, batch_size, context_hidden_size]
            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden,
                                                                        context_hidden)

        # Run for generation
        for j in range(self.config.n_sample_step):
            # context_outputs: [batch_size, context_hidden_size * direction]
            context_outputs = context_outputs.squeeze(1)
            decoder_init = self.context2decoder(context_outputs)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            # prediction: [batch_size, seq_len]
            prediction = prediction[:, 0, :]
            # length: [batch_size]
            length = [l[0] for l in length]
            length = to_var(torch.LongTensor(length))
            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction,
                                                           length)

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)

            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden,
                                                                        context_hidden)

        samples = torch.stack(samples, 1)
        return samples


class VHRED(nn.Module):
    def __init__(self, config):
        super(VHRED, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         config.rnn,
                                         config.num_layers,
                                         config.bidirectional,
                                         config.dropout)

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)
        self.context_encoder = layers.ContextRNN(context_input_size,
                                                 config.context_size,
                                                 config.rnn,
                                                 config.num_layers,
                                                 config.dropout)

        self.decoder = layers.DecoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.decoder_hidden_size,
                                         config.rnncell,
                                         config.num_layers,
                                         config.dropout,
                                         config.word_drop,
                                         config.max_unroll,
                                         config.sample,
                                         config.temperature,
                                         config.beam_size)


        self.context2decoder = layers.FeedForward(config.context_size + config.z_sent_size,
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1,
                                                  activation=config.activation)

        self.softplus = nn.Softplus()
        self.prior_h = layers.FeedForward(config.context_size,
                                          config.context_size,
                                          num_layers=2,
                                          hidden_size=config.context_size,
                                          activation=config.activation)
        self.prior_mu = nn.Linear(config.context_size,
                                  config.z_sent_size)
        self.prior_var = nn.Linear(config.context_size,
                                   config.z_sent_size)

        self.posterior_h = layers.FeedForward(config.encoder_hidden_size * self.encoder.num_directions * config.num_layers + config.context_size,
                                              config.context_size,
                                              num_layers=2,
                                              hidden_size=config.context_size,
                                              activation=config.activation)
        self.posterior_mu = nn.Linear(config.context_size,
                                      config.z_sent_size)
        self.posterior_var = nn.Linear(config.context_size,
                                       config.z_sent_size)
        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

        if config.bow:
            self.bow_h = layers.FeedForward(config.z_sent_size,
                                            config.decoder_hidden_size,
                                            num_layers=1,
                                            hidden_size=config.decoder_hidden_size,
                                            activation=config.activation)
            self.bow_predict = nn.Linear(config.decoder_hidden_size, config.vocab_size)

    def prior(self, context_outputs):
        # Context dependent prior
        h_prior = self.prior_h(context_outputs)
        mu_prior = self.prior_mu(h_prior)
        var_prior = self.softplus(self.prior_var(h_prior))
        return mu_prior, var_prior

    def posterior(self, context_outputs, encoder_hidden):
        h_posterior = self.posterior_h(torch.cat([context_outputs, encoder_hidden], 1))
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = self.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def compute_bow_loss(self, target_conversations):
        target_bow = np.stack([to_bow(sent, self.config.vocab_size) for conv in target_conversations for sent in conv], axis=0)
        target_bow = to_var(torch.FloatTensor(target_bow))
        bow_logits = self.bow_predict(self.bow_h(self.z_sent))
        bow_loss = bag_of_words_loss(bow_logits, target_bow)
        return bow_loss

    def forward(self, sentences, sentence_length,
                input_conversation_length, target_sentences, decode=False):
        """
        Args:
            sentences: (Variable, LongTensor) [num_sentences + batch_size, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        batch_size = input_conversation_length.size(0)
        num_sentences = sentences.size(0) - batch_size
        max_len = input_conversation_length.data.max().item()

        # encoder_outputs: [num_sentences + batch_size, max_source_length, hidden_size]
        # encoder_hidden: [num_layers * direction, num_sentences + batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(sentences,
                                                       sentence_length)

        # encoder_hidden: [num_sentences + batch_size, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(
            1, 0).contiguous().view(num_sentences + batch_size, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1] + 1)), 0)
        # encoder_hidden: [batch_size, max_len + 1, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l + 1), max_len + 1)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        # encoder_hidden_inference: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat(
            [encoder_hidden_inference[i, :l, :] for i, l in enumerate(input_conversation_length.data)])

        # encoder_hidden_input: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_input = encoder_hidden[:, :-1, :]

        # context_outputs: [batch_size, max_len, context_size]
        context_outputs, context_last_hidden = self.context_encoder(encoder_hidden_input,
                                                                    input_conversation_length)
        # flatten outputs
        # context_outputs: [num_sentences, context_size]
        context_outputs = torch.cat([context_outputs[i, :l, :]
                                     for i, l in enumerate(input_conversation_length.data)])

        mu_prior, var_prior = self.prior(context_outputs)
        eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))
        if not decode:
            mu_posterior, var_posterior = self.posterior(
                context_outputs, encoder_hidden_inference_flat)
            z_sent = mu_posterior + torch.sqrt(var_posterior) * eps
            log_q_zx = normal_logpdf(z_sent, mu_posterior, var_posterior).sum()

            log_p_z = normal_logpdf(z_sent, mu_prior, var_prior).sum()
            # kl_div: [num_sentneces]
            kl_div = normal_kl_div(mu_posterior, var_posterior,
                                    mu_prior, var_prior)
            kl_div = torch.sum(kl_div)
        else:
            z_sent = mu_prior + torch.sqrt(var_prior) * eps
            kl_div = None
            log_p_z = normal_logpdf(z_sent, mu_prior, var_prior).sum()
            log_q_zx = None

        self.z_sent = z_sent
        latent_context = torch.cat([context_outputs, z_sent], 1)
        decoder_init = self.context2decoder(latent_context)
        decoder_init = decoder_init.view(-1,
                                         self.decoder.num_layers,
                                         self.decoder.hidden_size)
        decoder_init = decoder_init.transpose(1, 0).contiguous()

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:

            decoder_outputs = self.decoder(target_sentences,
                                           init_h=decoder_init,
                                           decode=decode)

            return decoder_outputs, kl_div, log_p_z, log_q_zx

        else:
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)

            return prediction, kl_div, log_p_z, log_q_zx

    def generate(self, context, sentence_length, n_context):
        # context: [batch_size, n_context, seq_len]
        batch_size = context.size(0)
        # n_context = context.size(1)
        samples = []

        # Run for context
        context_hidden=None
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :],
                                                           sentence_length[:, i])

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            # context_outputs: [batch_size, 1, context_hidden_size * direction]
            # context_hidden: [num_layers * direction, batch_size, context_hidden_size]
            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden,
                                                                        context_hidden)

        # Run for generation
        for j in range(self.config.n_sample_step):
            # context_outputs: [batch_size, context_hidden_size * direction]
            context_outputs = context_outputs.squeeze(1)

            mu_prior, var_prior = self.prior(context_outputs)
            eps = to_var(torch.randn((batch_size, self.config.z_sent_size)))
            z_sent = mu_prior + torch.sqrt(var_prior) * eps

            latent_context = torch.cat([context_outputs, z_sent], 1)
            decoder_init = self.context2decoder(latent_context)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

            if self.config.sample:
                prediction = self.decoder(None, decoder_init)
                p = prediction.data.cpu().numpy()
                length = torch.from_numpy(np.where(p == EOS_ID)[1])
            else:
                prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
                # prediction: [batch_size, seq_len]
                prediction = prediction[:, 0, :]
                # length: [batch_size]
                length = [l[0] for l in length]
                length = to_var(torch.LongTensor(length))

            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction,
                                                           length)

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)

            context_outputs, context_hidden = self.context_encoder.step(encoder_hidden,
                                                                        context_hidden)

        samples = torch.stack(samples, 1)
        return samples


    
    
class VHCR(nn.Module):
    def __init__(self, config):
        super(VHCR, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         config.rnn,
                                         config.num_layers,
                                         config.bidirectional,
                                         config.dropout)

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions + config.z_conv_size)
        self.context_encoder = layers.ContextRNN(context_input_size,
                                                 config.context_size,
                                                 config.rnn,
                                                 config.num_layers,
                                                 config.dropout)

        self.unk_sent = nn.Parameter(torch.randn(context_input_size - config.z_conv_size))

        self.z_conv2context = layers.FeedForward(config.z_conv_size,
                                                 config.num_layers * config.context_size,
                                                 num_layers=1,
                                                 activation=config.activation)

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)
        self.context_inference = layers.ContextRNN(context_input_size,
                                                   config.context_size,
                                                   config.rnn,
                                                   config.num_layers,
                                                   config.dropout,
                                                   bidirectional=True)

        self.decoder = layers.DecoderRNN(config.vocab_size,
                                        config.embedding_size,
                                        config.decoder_hidden_size,
                                        config.rnncell,
                                        config.num_layers,
                                        config.dropout,
                                        config.word_drop,
                                        config.max_unroll,
                                        config.sample,
                                        config.temperature,
                                        config.beam_size)

        self.context2decoder = layers.FeedForward(config.context_size + config.z_sent_size + config.z_conv_size,
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1,
                                                  activation=config.activation)

        self.softplus = nn.Softplus()

        self.conv_posterior_h = layers.FeedForward(config.num_layers * self.context_inference.num_directions * config.context_size,
                                                    config.context_size,
                                                    num_layers=2,
                                                    hidden_size=config.context_size,
                                                    activation=config.activation)
        self.conv_posterior_mu = nn.Linear(config.context_size,
                                            config.z_conv_size)
        self.conv_posterior_var = nn.Linear(config.context_size,
                                             config.z_conv_size)

        self.sent_prior_h = layers.FeedForward(config.context_size + config.z_conv_size,
                                               config.context_size,
                                               num_layers=1,
                                               hidden_size=config.z_sent_size,
                                               activation=config.activation)
        self.sent_prior_mu = nn.Linear(config.context_size,
                                       config.z_sent_size)
        self.sent_prior_var = nn.Linear(config.context_size,
                                        config.z_sent_size)

        self.sent_posterior_h = layers.FeedForward(config.z_conv_size + config.encoder_hidden_size * self.encoder.num_directions * config.num_layers + config.context_size,
                                                   config.context_size,
                                                   num_layers=2,
                                                   hidden_size=config.context_size,
                                                   activation=config.activation)
        self.sent_posterior_mu = nn.Linear(config.context_size,
                                           config.z_sent_size)
        self.sent_posterior_var = nn.Linear(config.context_size,
                                            config.z_sent_size)

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def conv_prior(self):
        # Standard gaussian prior
        return to_var(torch.FloatTensor([0.0])), to_var(torch.FloatTensor([1.0]))

    def conv_posterior(self, context_inference_hidden):
        h_posterior = self.conv_posterior_h(context_inference_hidden)
        mu_posterior = self.conv_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.conv_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def sent_prior(self, context_outputs, z_conv):
        # Context dependent prior
        h_prior = self.sent_prior_h(torch.cat([context_outputs, z_conv], dim=1))
        mu_prior = self.sent_prior_mu(h_prior)
        var_prior = self.softplus(self.sent_prior_var(h_prior))
        return mu_prior, var_prior

    def sent_posterior(self, context_outputs, encoder_hidden, z_conv):
        h_posterior = self.sent_posterior_h(torch.cat([context_outputs, encoder_hidden, z_conv], 1))
        mu_posterior = self.sent_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.sent_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def forward(self, sentences, sentence_length,
                input_conversation_length, target_sentences, decode=False):
        """
        Args:
            sentences: (Variable, LongTensor) [num_sentences + batch_size, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        batch_size = input_conversation_length.size(0)
        num_sentences = sentences.size(0) - batch_size
        max_len = input_conversation_length.data.max().item()

        # encoder_outputs: [num_sentences + batch_size, max_source_length, hidden_size]
        # encoder_hidden: [num_layers * direction, num_sentences + batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(sentences,
                                                       sentence_length)

        # encoder_hidden: [num_sentences + batch_size, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(
            1, 0).contiguous().view(num_sentences + batch_size, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1] + 1)), 0)
        # encoder_hidden: [batch_size, max_len + 1, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l + 1), max_len + 1)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        # encoder_hidden_inference: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat(
            [encoder_hidden_inference[i, :l, :] for i, l in enumerate(input_conversation_length.data)])

        # encoder_hidden_input: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_input = encoder_hidden[:, :-1, :]

        # Standard Gaussian prior
        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))
        conv_mu_prior, conv_var_prior = self.conv_prior()

        if not decode:
            if self.config.sentence_drop > 0.0:
                indices = np.where(np.random.rand(max_len) < self.config.sentence_drop)[0]
                if len(indices) > 0:
                    encoder_hidden_input[:, indices, :] = self.unk_sent

            # context_inference_outputs: [batch_size, max_len, num_directions * context_size]
            # context_inference_hidden: [num_layers * num_directions, batch_size, hidden_size]
            context_inference_outputs, context_inference_hidden = self.context_inference(encoder_hidden,
                                                                                            input_conversation_length + 1)

            # context_inference_hidden: [batch_size, num_layers * num_directions * hidden_size]
            context_inference_hidden = context_inference_hidden.transpose(
                1, 0).contiguous().view(batch_size, -1)
            conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden)
            z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior) * conv_eps
            log_q_zx_conv = normal_logpdf(z_conv, conv_mu_posterior, conv_var_posterior).sum()

            log_p_z_conv = normal_logpdf(z_conv, conv_mu_prior, conv_var_prior).sum()
            kl_div_conv = normal_kl_div(conv_mu_posterior, conv_var_posterior,
                                            conv_mu_prior, conv_var_prior).sum()

            context_init = self.z_conv2context(z_conv).view(
                self.config.num_layers, batch_size, self.config.context_size)

            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(
                1)).expand(z_conv.size(0), max_len, z_conv.size(1))
            context_outputs, context_last_hidden = self.context_encoder(
                torch.cat([encoder_hidden_input, z_conv_expand], 2),
                input_conversation_length,
                hidden=context_init)

            # flatten outputs
            # context_outputs: [num_sentences, context_size]
            context_outputs = torch.cat([context_outputs[i, :l, :]
                                         for i, l in enumerate(input_conversation_length.data)])

            z_conv_flat = torch.cat(
                [z_conv_expand[i, :l, :] for i, l in enumerate(input_conversation_length.data)])
            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs, z_conv_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))

            sent_mu_posterior, sent_var_posterior = self.sent_posterior(
                context_outputs, encoder_hidden_inference_flat, z_conv_flat)
            z_sent = sent_mu_posterior + torch.sqrt(sent_var_posterior) * eps
            log_q_zx_sent = normal_logpdf(z_sent, sent_mu_posterior, sent_var_posterior).sum()

            log_p_z_sent = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior).sum()
            # kl_div: [num_sentences]
            kl_div_sent = normal_kl_div(sent_mu_posterior, sent_var_posterior,
                                        sent_mu_prior, sent_var_prior).sum()

            kl_div = kl_div_conv + kl_div_sent
            log_q_zx = log_q_zx_conv + log_q_zx_sent
            log_p_z = log_p_z_conv + log_p_z_sent
        else:
            z_conv = conv_mu_prior + torch.sqrt(conv_var_prior) * conv_eps
            context_init = self.z_conv2context(z_conv).view(
                self.config.num_layers, batch_size, self.config.context_size)

            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(
                1)).expand(z_conv.size(0), max_len, z_conv.size(1))
            # context_outputs: [batch_size, max_len, context_size]
            context_outputs, context_last_hidden = self.context_encoder(
                torch.cat([encoder_hidden_input, z_conv_expand], 2),
                input_conversation_length,
                hidden=context_init)
            # flatten outputs
            # context_outputs: [num_sentences, context_size]
            context_outputs = torch.cat([context_outputs[i, :l, :]
                                         for i, l in enumerate(input_conversation_length.data)])


            z_conv_flat = torch.cat(
                [z_conv_expand[i, :l, :] for i, l in enumerate(input_conversation_length.data)])
            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs, z_conv_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))

            z_sent = sent_mu_prior + torch.sqrt(sent_var_prior) * eps
            kl_div = None
            log_p_z = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior).sum()
            log_p_z += normal_logpdf(z_conv, conv_mu_prior, conv_var_prior).sum()
            log_q_zx = None

        # expand z_conv to all associated sentences
        z_conv = torch.cat([z.view(1, -1).expand(m.item(), self.config.z_conv_size)
                             for z, m in zip(z_conv, input_conversation_length)])

        # latent_context: [num_sentences, context_size + z_sent_size +
        # z_conv_size]
        latent_context = torch.cat([context_outputs, z_sent, z_conv], 1)
        decoder_init = self.context2decoder(latent_context)
        decoder_init = decoder_init.view(-1,
                                         self.decoder.num_layers,
                                         self.decoder.hidden_size)
        decoder_init = decoder_init.transpose(1, 0).contiguous()

        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        if not decode:
            decoder_outputs = self.decoder(target_sentences,
                                            init_h=decoder_init,
                                            decode=decode)
            return decoder_outputs, kl_div, log_p_z, log_q_zx

        else:
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            return prediction, kl_div, log_p_z, log_q_zx

    def generate(self, context, sentence_length, n_context):
        # context: [batch_size, n_context, seq_len]
        batch_size = context.size(0)
        # n_context = context.size(1)
        samples = []

        # Run for context

        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))
        # conv_mu_prior, conv_var_prior = self.conv_prior()
        # z_conv = conv_mu_prior + torch.sqrt(conv_var_prior) * conv_eps

        encoder_hidden_list = []
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :],
                                                           sentence_length[:, i])

            # encoder_hidden: [batch_size, num_layers * direction * hidden_size]
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            encoder_hidden_list.append(encoder_hidden)

        encoder_hidden = torch.stack(encoder_hidden_list, 1)
        context_inference_outputs, context_inference_hidden = self.context_inference(encoder_hidden,
                                                                                     to_var(torch.LongTensor([n_context] * batch_size)))
        context_inference_hidden = context_inference_hidden.transpose(
            1, 0).contiguous().view(batch_size, -1)
        conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden)
        z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior) * conv_eps

        context_init = self.z_conv2context(z_conv).view(
            self.config.num_layers, batch_size, self.config.context_size)

        context_hidden = context_init
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :],
                                                           sentence_length[:, i])

            # encoder_hidden: [batch_size, num_layers * direction *
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            encoder_hidden_list.append(encoder_hidden)
            # context_outputs: [batch_size, 1, context_hidden_size * direction]
            # context_hidden: [num_layers * direction, batch_size, context_hidden_size]
            context_outputs, context_hidden = self.context_encoder.step(torch.cat([encoder_hidden, z_conv], 1),
                                                                        context_hidden)

        # Run for generation
        for j in range(self.config.n_sample_step):
            # context_outputs: [batch_size, context_hidden_size * direction]
            context_outputs = context_outputs.squeeze(1)

            mu_prior, var_prior = self.sent_prior(context_outputs, z_conv)
            eps = to_var(torch.randn((batch_size, self.config.z_sent_size)))
            z_sent = mu_prior + torch.sqrt(var_prior) * eps

            latent_context = torch.cat([context_outputs, z_sent, z_conv], 1)
            decoder_init = self.context2decoder(latent_context)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

            if self.config.sample:
                prediction = self.decoder(None, decoder_init, decode=True)
                p = prediction.data.cpu().numpy()
                length = torch.from_numpy(np.where(p == EOS_ID)[1])
            else:
                prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
                # prediction: [batch_size, seq_len]
                prediction = prediction[:, 0, :]
                # length: [batch_size]
                length = [l[0] for l in length]
                length = to_var(torch.LongTensor(length))

            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction,
                                                           length)

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)

            context_outputs, context_hidden = self.context_encoder.step(torch.cat([encoder_hidden, z_conv], 1),
                                                                        context_hidden)

        samples = torch.stack(samples, 1)
        return samples    
    
    
        
    

class VHCR_new(nn.Module):
    def __init__(self, config):
        super(VHCR, self).__init__()

        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size,
                                         config.embedding_size,
                                         config.encoder_hidden_size,
                                         config.rnn,
                                         config.num_layers,
                                         config.bidirectional,
                                         config.dropout)

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions + config.z_conv_size+config.z_ctx_size) ## add z_ctx
        self.context_encoder = layers.ContextRNN(context_input_size,
                                                 config.context_size,
                                                 config.rnn,
                                                 config.num_layers,
                                                 config.dropout)

        self.unk_sent = nn.Parameter(torch.randn(context_input_size - config.z_conv_size-config.z_ctx_size)) ##omit dimension config.z_ctx_size

        self.z_conv2context = layers.FeedForward(config.z_conv_size,
                                                 config.num_layers * config.context_size,
                                                 num_layers=1,
                                                 activation=config.activation)
        
        """edit here"""
        self.context2context_first = layers.Context2Context(config.context_size*2+config.z_conv_size,
                                                            config.context_size)
        self.context2context = layers.Context2Context(config.context_size*2
                                                      +config.z_conv_size+config.z_ctx_size,
                                                            config.context_size)
        """stop here"""

        context_input_size = (config.num_layers
                              * config.encoder_hidden_size
                              * self.encoder.num_directions)  ##infer z_conv
        self.context_inference = layers.ContextRNN(context_input_size,   
                                                   config.context_size,
                                                   config.rnn,
                                                   config.num_layers,
                                                   config.dropout,
                                                   bidirectional=True)
        #Change bidirectional to false to make hidden_size same when inferring in ctx_z
        self.ctx_inference = layers.ContextRNN(context_input_size,   
                                                   config.context_size,
                                                   config.rnn,
                                                   config.num_layers,
                                                   config.dropout)        
 

        self.decoder = layers.DecoderRNN(config.vocab_size,
                                        config.embedding_size,
                                        config.decoder_hidden_size,
                                        config.rnncell,
                                        config.num_layers,
                                        config.dropout,
                                        config.word_drop,
                                        config.max_unroll,
                                        config.sample,
                                        config.temperature,
                                        config.beam_size)

        self.context2decoder = layers.FeedForward(config.context_size + config.z_sent_size + config.z_conv_size+config.z_ctx_size,  ##add config.z_ctx_size
                                                  config.num_layers * config.decoder_hidden_size,
                                                  num_layers=1,
                                                  activation=config.activation)

        self.softplus = nn.Softplus()

        self.conv_posterior_h = layers.FeedForward(config.num_layers * self.context_inference.num_directions * config.context_size,
                                                    config.context_size,
                                                    num_layers=2,
                                                    hidden_size=config.context_size,
                                                    activation=config.activation)
        self.conv_posterior_mu = nn.Linear(config.context_size,
                                            config.z_conv_size)
        self.conv_posterior_var = nn.Linear(config.context_size,
                                             config.z_conv_size)

        #z_sen do add z_ctx or not?
        self.sent_prior_h = layers.FeedForward(config.context_size + config.z_conv_size+config.z_ctx_size,  #z_sen_prior(h_ctx,z_conv,z_ctx)
                                               config.context_size,
                                               num_layers=1,
                                               hidden_size=config.z_sent_size,
                                               activation=config.activation)
        self.sent_prior_mu = nn.Linear(config.context_size,
                                       config.z_sent_size)
        self.sent_prior_var = nn.Linear(config.context_size,
                                        config.z_sent_size)

        self.sent_posterior_h = layers.FeedForward(config.z_conv_size+config.z_ctx_size+config.encoder_hidden_size * self.encoder.num_directions * config.num_layers + config.context_size,
                                                   config.context_size,  ##z_sen_posterior (h_ctx,z_conv,z_ctx,x_encode)
                                                   num_layers=2,
                                                   hidden_size=config.context_size,
                                                   activation=config.activation)
        self.sent_posterior_mu = nn.Linear(config.context_size,
                                           config.z_sent_size)
        self.sent_posterior_var = nn.Linear(config.context_size,
                                            config.z_sent_size)
        ##z_ctx_prior
        self.ctx_prior_h = layers.FeedForward(config.z_conv_size+config.context_size,
                                                 config.context_size,
                                                 num_layers=1,
                                                 hidden_size=config.context_size,  ##use config.context_size or config.z_ctx_size?
                                                 activation=config.activation)
        self.ctx_prior_mu = nn.Linear(config.context_size,
                                      config.z_ctx_size)
        self.ctx_prior_var = nn.Linear(config.context_size,
                                       config.z_ctx_size)
        ##z_ctx_posterior
        self.ctx_posterior_h = layers.FeedForward(config.z_conv_size+config.context_size+config.num_layers * self.context_inference.num_directions * config.context_size,
                                                    config.context_size,
                                                    num_layers=2,
                                                    hidden_size=config.context_size,
                                                    activation=config.activation)
        ##context_h is based on x(<t),z_conv,h_context(t) but how to only use x(<t)  the expression above just like z_conv using x
        self.ctx_posterior_mu = nn.Linear(config.context_size,
                                         config.z_ctx_size)
        self.ctx_posterior_var = nn.Linear(config.context_size,
                                         config.z_ctx_size)
        

        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def conv_prior(self):
        # Standard gaussian prior
        return to_var(torch.FloatTensor([0.0])), to_var(torch.FloatTensor([1.0]))

    def conv_posterior(self, context_inference_hidden):
        h_posterior = self.conv_posterior_h(context_inference_hidden)
        mu_posterior = self.conv_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.conv_posterior_var(h_posterior))
        return mu_posterior, var_posterior
    ##ctx_prior
    def ctx_prior(self,context_outputs,z_conv):
        h_prior = self.ctx_prior_h(torch.cat([context_outputs, z_conv], dim=1))
        mu_prior = self.ctx_prior_mu(h_prior)
        var_prior = self.softplus(self.ctx_prior_var(h_prior))
        return mu_prior, var_prior
    #ctx_posterior
    def ctx_posterior(self,ctx_inference_hidden,context_outputs,z_conv):
        h_posterior = self.ctx_posterior_h(torch.cat([context_outputs,ctx_inference_hidden,z_conv],1))
        mu_posterior = self.ctx_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.ctx_posterior_var(h_posterior))
        return mu_posterior,var_posterior

    #do I need to add z_ctx in sen_prior and sen_posterior?
    def sent_prior(self, context_outputs, z_conv,z_ctx):
        # Context dependent prior
        h_prior = self.sent_prior_h(torch.cat([context_outputs, z_conv, z_ctx], dim=1))
        mu_prior = self.sent_prior_mu(h_prior)
        var_prior = self.softplus(self.sent_prior_var(h_prior))
        return mu_prior, var_prior

    def sent_posterior(self, context_outputs, encoder_hidden, z_conv,z_ctx):
        h_posterior = self.sent_posterior_h(torch.cat([context_outputs, encoder_hidden, z_conv, z_ctx], 1))
        mu_posterior = self.sent_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.sent_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def forward(self, sentences, sentence_length,
                input_conversation_length, target_sentences, decode=False):
        """
        Args:
            sentences: (Variable, LongTensor) [num_sentences + batch_size, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        #  num_sentences = sentences.size(0) - batch_size num_sentence is the input sentences
        batch_size = input_conversation_length.size(0)
        num_sentences = sentences.size(0) - batch_size
        max_len = input_conversation_length.data.max().item()
        
        # encoder_outputs: [num_sentences + batch_size, max_source_length, hidden_size]
        # encoder_hidden: [num_layers * direction, num_sentences + batch_size, hidden_size]
        encoder_outputs, encoder_hidden = self.encoder(sentences,
                                                       sentence_length)

        # encoder_hidden: [num_sentences + batch_size, num_layers * direction * hidden_size]
        encoder_hidden = encoder_hidden.transpose(
            1, 0).contiguous().view(num_sentences + batch_size, -1)

        # pad and pack encoder_hidden
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.data.new(1).zero_()),
                                        input_conversation_length[:-1] + 1)), 0)
        # encoder_hidden: [batch_size, max_len + 1, num_layers * direction * hidden_size]
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l + 1), max_len + 1)
                                      for s, l in zip(start.data.tolist(),
                                                      input_conversation_length.data.tolist())], 0)

        # encoder_hidden_inference: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat(
            [encoder_hidden_inference[i, :l, :] for i, l in enumerate(input_conversation_length.data)])

        # encoder_hidden_input: [batch_size, max_len, num_layers * direction * hidden_size]
        encoder_hidden_input = encoder_hidden[:, :-1, :]

        # Standard Gaussian prior
        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))
        conv_mu_prior, conv_var_prior = self.conv_prior()

        if not decode:
            if self.config.sentence_drop > 0.0:
                indices = np.where(np.random.rand(max_len) < self.config.sentence_drop)[0]
                if len(indices) > 0:
                    encoder_hidden_input[:, indices, :] = self.unk_sent
            

            # context_inference_outputs: [batch_size, max_len, num_directions * context_size]
            # context_inference_hidden: [num_layers * num_directions, batch_size, hidden_size]
            context_inference_outputs, context_inference_hidden = self.context_inference(encoder_hidden,
                                                                                            input_conversation_length + 1)

            # context_inference_hidden: [batch_size, num_layers * num_directions * hidden_size]
            context_inference_hidden = context_inference_hidden.transpose(
                1, 0).contiguous().view(batch_size, -1)
            conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden)
            z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior) * conv_eps
            log_q_zx_conv = normal_logpdf(z_conv, conv_mu_posterior, conv_var_posterior).sum()

            log_p_z_conv = normal_logpdf(z_conv, conv_mu_prior, conv_var_prior).sum()
            kl_div_conv = normal_kl_div(conv_mu_posterior, conv_var_posterior,
                                            conv_mu_prior, conv_var_prior).sum()

            context_init = self.z_conv2context(z_conv).view(
                self.config.num_layers, batch_size, self.config.context_size)

            #expand z_conv to batch_size, max_len, z_conv_size then every sentence can use z_conv
            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(
                1)).expand(z_conv.size(0), max_len, z_conv.size(1))
            
            """code for context"""
            context_outputs = []
            z_ctx = []
            ctx_mu_prior = []
            ctx_var_prior = []
            ctx_mu_posterior = []
            ctx_var_posterior = []
            ctx_eps = to_var(torch.randn((batch_size, self.config.z_ctx_size)))            
            # context_inference_outputs: [batch_size, max_len, num_directions * context_size]
            # context_inference_hidden: [num_layers * num_directions, batch_size, hidden_size]
            ctx_inference_outputs, ctx_inference_hidden = self.context_inference(encoder_hidden,
                                                                                         input_conversation_length + 1)
            context_init1 = context_init.squeeze(0)
            for i_ctx in range(max_len):
                if i_ctx==0:
                    context_cell_input = torch.cat([encoder_hidden_input[:,i_ctx,:], z_conv_expand[:,i_ctx,:]], 1)                 
                    context_h_cell = self.context2context_first(
                        context_cell_input,
                        hidden=context_init1)
                else:
                    context_cell_input = torch.cat([encoder_hidden_input[:,i_ctx,:],  z_conv_expand[:,i_ctx,:],z_ctx_cell],1)     
                    context_h_cell = self.context2context(
                        context_cell_input,
                        hidden=context_h_cell)
                #z_ctx_cell: [batch_size,num_directions * context_size]
                ctx_cell_mu_prior,ctx_cell_var_prior = self.ctx_prior(context_h_cell,z_conv_expand[:,i_ctx,:])               
                ctx_cell_mu_posterior,ctx_cell_var_posterior = self.ctx_posterior(ctx_inference_outputs[:,i_ctx,:],context_h_cell,z_conv_expand[:,i_ctx,:])
                z_ctx_cell = ctx_cell_mu_posterior + torch.sqrt(ctx_cell_var_posterior) * ctx_eps
                z_ctx.append(z_ctx_cell)
                ctx_mu_prior.append(ctx_cell_mu_prior)
                ctx_var_prior.append(ctx_cell_var_prior)
                ctx_mu_posterior.append(ctx_cell_mu_posterior)
                ctx_var_posterior.append(ctx_cell_var_posterior)
                context_outputs.append(context_h_cell)
            #context_outputs: [batch_size,max_len,context_size]
            #z_ctx:[batch_size,max_len,context_size]
            context_outputs = torch.stack(context_outputs,1)
            ctx_mu_prior = torch.stack(ctx_mu_prior,1)
            ctx_var_prior = torch.stack(ctx_var_prior,1)
            ctx_mu_posterior = torch.stack(ctx_mu_posterior,1)
            ctx_var_posterior = torch.stack(ctx_var_posterior,1)
            z_ctx = torch.stack(z_ctx,1)

            
            """stop here"""

            # flatten outputs
            # context_outputs: [num_sentences, context_size]
            context_outputs = torch.cat([context_outputs[i, :l, :]
                                         for i, l in enumerate(input_conversation_length.data)])

            z_conv_flat = torch.cat(
                [z_conv_expand[i, :l, :] for i, l in enumerate(input_conversation_length.data)])
            ##z_ctx faltten and its mu and var
            """code edit here"""
            z_ctx_flat = torch.cat(
                [z_ctx[i, :l, :] for i, l in enumerate(input_conversation_length.data)])  
            ctx_mu_posterior_flat = torch.cat(
                [ctx_mu_posterior[i, :l, :] for i, l in enumerate(input_conversation_length.data)])  

            ctx_var_posterior_flat = torch.cat(
                [ctx_var_posterior[i, :l, :] for i, l in enumerate(input_conversation_length.data)])  
            ctx_mu_prior_flat = torch.cat(
                [ctx_mu_prior[i, :l, :] for i, l in enumerate(input_conversation_length.data)])  
            ctx_var_prior_flat = torch.cat(
                [ctx_var_prior[i, :l, :] for i, l in enumerate(input_conversation_length.data)])  
            log_q_zx_ctx = normal_logpdf(z_ctx_flat, ctx_mu_posterior_flat, ctx_var_posterior_flat).sum()
            log_p_z_ctx = normal_logpdf(z_ctx_flat, ctx_mu_prior_flat, ctx_var_prior_flat).sum()
            kl_div_ctx = normal_kl_div(ctx_mu_posterior_flat, ctx_var_posterior_flat,
                                        ctx_mu_prior_flat, ctx_var_prior_flat).sum()     
            
            """stop here"""
            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs, z_conv_flat,z_ctx_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))

            sent_mu_posterior, sent_var_posterior = self.sent_posterior(
                context_outputs, encoder_hidden_inference_flat, z_conv_flat,z_ctx_flat)
            z_sent = sent_mu_posterior + torch.sqrt(sent_var_posterior) * eps
            log_q_zx_sent = normal_logpdf(z_sent, sent_mu_posterior, sent_var_posterior).sum()

            log_p_z_sent = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior).sum()
            # kl_div: [num_sentences]
            kl_div_sent = normal_kl_div(sent_mu_posterior, sent_var_posterior,
                                        sent_mu_prior, sent_var_prior).sum()

            kl_div = kl_div_conv + kl_div_sent+kl_div_ctx
            log_q_zx = log_q_zx_conv + log_q_zx_sent + log_q_zx_ctx
            log_p_z = log_p_z_conv + log_p_z_sent + log_p_z_ctx
        else:
            z_conv = conv_mu_prior + torch.sqrt(conv_var_prior) * conv_eps
            context_init = self.z_conv2context(z_conv).view(
                self.config.num_layers, batch_size, self.config.context_size)

            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(
                1)).expand(z_conv.size(0), max_len, z_conv.size(1))
            # context_outputs: [batch_size, max_len, context_size]
            
            
            """code for context how to dropout? need dropout?"""
            context_outputs = []
            z_ctx = []
            ctx_mu_prior = []
            ctx_var_prior = []
            ctx_eps = to_var(torch.randn((batch_size, self.config.z_ctx_size)))            
            # context_inference_outputs: [batch_size, max_len, num_directions * context_size]
            # context_inference_hidden: [num_layers * num_directions, batch_size, hidden_size]
            ctx_inference_outputs, ctx_inference_hidden = self.context_inference(encoder_hidden,
                                                                                         input_conversation_length + 1)

            context_init1 = context_init.squeeze(0)
            for i_ctx in range(max_len):
                if i_ctx==0:
                    context_cell_input = torch.cat([encoder_hidden_input[:,i_ctx,:], z_conv_expand[:,i_ctx,:]], 1)                 
                    context_h_cell = self.context2context_first(
                        context_cell_input,
                        hidden=context_init1)
                else:
                    context_cell_input = torch.cat([encoder_hidden_input[:,i_ctx,:],  z_conv_expand[:,i_ctx,:],z_ctx_cell],1)     
                    context_h_cell = self.context2context(
                        context_cell_input,
                        hidden=context_h_cell)
                #z_ctx_cell: [batch_size,num_directions * context_size]
                ctx_cell_mu_prior,ctx_cell_var_prior = self.ctx_prior(context_h_cell,z_conv_expand[:,i_ctx,:])               
                z_ctx_cell = ctx_cell_mu_prior + torch.sqrt(ctx_cell_var_prior) * ctx_eps
                z_ctx.append(z_ctx_cell)
                ctx_mu_prior.append(ctx_cell_mu_prior)
                ctx_var_prior.append(ctx_cell_var_prior)
                context_outputs.append(context_h_cell)
            #context_outputs: [batch_size,max_len,context_size]
            #z_ctx:[batch_size,max_len,context_size]
            context_outputs = torch.stack(context_outputs,1)
            ctx_mu_prior = torch.stack(ctx_mu_prior,1)
            ctx_var_prior = torch.stack(ctx_var_prior,1)  
            z_ctx = torch.stack(z_ctx,1)
       
            """stop here"""
            # flatten outputs
            # context_outputs: [num_sentences, context_size]
            context_outputs = torch.cat([context_outputs[i, :l, :]
                                         for i, l in enumerate(input_conversation_length.data)])


            z_conv_flat = torch.cat(
                [z_conv_expand[i, :l, :] for i, l in enumerate(input_conversation_length.data)])
            """code edit here"""
            z_ctx_flat = torch.cat(
                [z_ctx[i, :l, :] for i, l in enumerate(input_conversation_length.data)])  
            ctx_mu_prior_flat = torch.cat(
                [ctx_mu_prior[i, :l, :] for i, l in enumerate(input_conversation_length.data)])  
            ctx_var_prior_flat = torch.cat(
                [ctx_var_prior[i, :l, :] for i, l in enumerate(input_conversation_length.data)])  
            """stop here"""

            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs, z_conv_flat,z_ctx_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))
            z_sent = sent_mu_prior + torch.sqrt(sent_var_prior) * eps
            kl_div = None
            log_p_z = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior).sum()
            log_p_z += normal_logpdf(z_conv_flat, conv_mu_prior, conv_var_prior).sum()
            log_p_z += normal_logpdf(z_ctx_flat, ctx_mu_prior_flat, ctx_var_prior_flat).sum()
            log_q_zx = None

        # expand z_conv to all associated sentences
        #expand z_ctx        
#        z_ctx = torch.cat([z.view(1, -1).expand(m.item(), self.config.z_ctx_size)
#                             for z, m in zip(z_ctx, input_conversation_length)])
        z_conv = torch.cat([z.view(1, -1).expand(m.item(), self.config.z_conv_size)
                             for z, m in zip(z_conv, input_conversation_length)])

        # latent_context: [num_sentences, context_size + z_sent_size +
        # z_conv_size+z_ctx]
        latent_context = torch.cat([context_outputs, z_sent, z_conv,z_ctx_flat], 1)
        decoder_init = self.context2decoder(latent_context)
        decoder_init = decoder_init.view(-1,
                                         self.decoder.num_layers,
                                         self.decoder.hidden_size)
        decoder_init = decoder_init.transpose(1, 0).contiguous()
        #decoder_init: [num_layers,batch_size,hidden_size]
        # train: [batch_size, seq_len, vocab_size]
        # eval: [batch_size, seq_len]
        
        """edit here"""
        if not decode:
            decoder_outputs = self.decoder(target_sentences,
                                            init_h=decoder_init,
                                            decode=decode)
            return decoder_outputs, kl_div, log_p_z, log_q_zx
        ## decoder_outputs  [batch_size, max_target_len, vocab_size]
        else:
            # prediction: [batch_size, beam_size, max_unroll]
            prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
            return prediction, kl_div, log_p_z, log_q_zx
        
        
        
    def generate(self, context, sentence_length, n_context):
        # context: [batch_size, n_context, seq_len]
        batch_size = context.size(0)
        # n_context = context.size(1)
        samples = []

        # Run for context

        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))
        # conv_mu_prior, conv_var_prior = self.conv_prior()
        # z_conv = conv_mu_prior + torch.sqrt(conv_var_prior) * conv_eps

        encoder_hidden_list = []
        for i in range(n_context):
            # encoder_outputs: [batch_size, seq_len, hidden_size * direction]
            # encoder_hidden: [num_layers * direction, batch_size, hidden_size]
            """encoder use nn.GRU which need a 3 dimension input context[:, i, :] ? two dimension right? """
            encoder_outputs, encoder_hidden = self.encoder(context[:, i, :],
                                                           sentence_length[:, i])

            # encoder_hidden: [batch_size, num_layers * direction * hidden_size]
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            encoder_hidden_list.append(encoder_hidden)

        ##encoder_hidden:[batch_size, n_context, hidden_size * direction]
        encoder_hidden = torch.stack(encoder_hidden_list, 1)
        context_inference_outputs, context_inference_hidden = self.context_inference(encoder_hidden,
                                                                                     to_var(torch.LongTensor([n_context] * batch_size)))
        context_inference_hidden = context_inference_hidden.transpose(
            1, 0).contiguous().view(batch_size, -1)
        conv_mu_posterior, conv_var_posterior = self.conv_posterior(context_inference_hidden)
        z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior) * conv_eps

        context_init = self.z_conv2context(z_conv).view(
            self.config.num_layers, batch_size, self.config.context_size)
        #context_hidden = context_init
        
        """edit here"""    
        z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(
            1)).expand(z_conv.size(0), n_context, z_conv.size(1))
#         context_outputs = []
#         z_ctx = []
#         ctx_mu_prior = []
#         ctx_var_prior = []
#         ctx_mu_posterior = []
#         ctx_var_posterior = []
        ctx_eps = to_var(torch.randn((batch_size, self.config.z_ctx_size)))            
        # context_inference_outputs: [batch_size, max_len, num_directions * context_size]
        # context_inference_hidden: [num_layers * num_directions, batch_size, hidden_size]
        ctx_inference_outputs, ctx_inference_hidden = self.context_inference(encoder_hidden,
                                                                                         to_var(torch.LongTensor([n_context] * batch_size))) ##edit here
        context_init1 = context_init.squeeze(0)
        for i_ctx in range(n_context):
            if i_ctx==0:
                context_cell_input = torch.cat([encoder_hidden[:,i_ctx,:], z_conv_expand[:,i_ctx,:]], 1)                 
                context_h_cell = self.context2context_first(
                    context_cell_input,
                    hidden=context_init1)
            else:
                context_cell_input = torch.cat([encoder_hidden[:,i_ctx,:],  z_conv_expand[:,i_ctx,:],z_ctx_cell],1)     
                context_h_cell = self.context2context(
                    context_cell_input,
                    hidden=context_h_cell)
            #z_ctx_cell: [batch_size,num_directions * context_size]
#               ctx_cell_mu_prior,ctx_cell_var_prior = self.ctx_prior(context_h_cell,z_conv_expand[:,i_ctx,:])               
            ctx_cell_mu_posterior,ctx_cell_var_posterior = self.ctx_posterior(ctx_inference_outputs[:,i_ctx,:],context_h_cell,z_conv_expand[:,i_ctx,:])
            z_ctx_cell = ctx_cell_mu_posterior + torch.sqrt(ctx_cell_var_posterior) * ctx_eps
#               z_ctx.append(z_ctx_cell)
#               ctx_mu_prior.append(ctx_cell_mu_prior)
#               ctx_var_prior.append(ctx_cell_var_prior)
#               ctx_mu_posterior.append(ctx_cell_mu_posterior)
#               ctx_var_posterior.append(ctx_cell_var_posterior)
#               context_outputs.append(context_h_cell)
            #context_outputs: [batch_size,n_context,context_size]
            #z_ctx:[batch_size,n_context,context_size]
        context_outputs = context_h_cell
        z_ctx = z_ctx_cell
#             ctx_mu_prior = torch.stack(ctx_mu_prior,1)
#             ctx_var_prior = torch.stack(ctx_var_prior,1)
#             ctx_mu_posterior = torch.stack(ctx_mu_posterior,1)
#             ctx_var_posterior = torch.stack(ctx_var_posterior,1)
#             z_ctx = torch.stack(z_ctx,1)

            
        """stop here"""

                    
        # Run for generation
        for j in range(self.config.n_sample_step):
            # context_outputs: [batch_size, context_hidden_size * direction]
            context_outputs = context_outputs.squeeze(1)

            mu_prior, var_prior = self.sent_prior(context_outputs, z_conv,z_ctx)
            eps = to_var(torch.randn((batch_size, self.config.z_sent_size)))
            z_sent = mu_prior + torch.sqrt(var_prior) * eps

            latent_context = torch.cat([context_outputs, z_sent, z_conv,z_ctx], 1)
            decoder_init = self.context2decoder(latent_context)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.decoder.hidden_size)

            if self.config.sample:
                prediction = self.decoder(None, decoder_init, decode=True)
                p = prediction.data.cpu().numpy()
                length = torch.from_numpy(np.where(p == EOS_ID)[1])
            else:
                prediction, final_score, length = self.decoder.beam_decode(init_h=decoder_init)
                # prediction: [batch_size, seq_len]
                prediction = prediction[:, 0, :]
                # length: [batch_size]
                length = [l[0] for l in length]
                length = to_var(torch.LongTensor(length))

            samples.append(prediction)

            encoder_outputs, encoder_hidden = self.encoder(prediction,
                                                           length)

            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(batch_size, -1)
            
            """edit here"""
            context_cell_input = torch.cat([encoder_hidden,z_conv,z_ctx],1)
            context_h_cell = self.context2context(
            context_cell_input,
            hidden = context_outputs)
            ctx_inference_outputs, ctx_inference_hidden = self.context_inference(encoder_hidden,
                                                                                 to_var(torch.LongTensor(batch_size))) 
            ctx_inference_hidden = ctx_inference_hidden.transpose(
                1, 0).contiguous().view(batch_size, -1)         
            ctx_cell_mu_posterior,ctx_cell_var_posterior = self.ctx_posterior(ctx_inference_hidden,context_h_cell,z_conv)
            z_ctx = ctx_cell_mu_posterior + torch.sqrt(ctx_cell_var_posterior) * ctx_eps
            context_outputs = context_h_cell
            """END EDIT"""

        samples = torch.stack(samples, 1)
        return samples



