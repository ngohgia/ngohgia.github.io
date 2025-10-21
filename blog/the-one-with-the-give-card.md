---
layout: page
title: On being stupid
description: Being stupid is a feature, not a bug
image: /assets/images/the-one-with-the-give-card/give-card_banner.jpeg
permalink: /the-one-with-the-give-card
---

<div style="text-align: center; padding-bottom: 10px">Originally posted on <a href="https://giveasia.substack.com/p/the-one-with-the-give-card" target="_blank">Give.Asia blog</a></div>

> Or when my customer support (almost) went wrong

<img style="width: 100%" src="/assets/images/the-one-with-the-give-card/give-card_banner.jpeg">
> No AI-generated images this week. Just plain old-style [text generator](https://www.thewordfinder.com/friends-font-generator/) :P

I read every question that our users write in via our customer support. Admittedly, this is a new habit that I only started doing a couple of months ago. At some point, I was told and have also felt myself that I have been quite out of touch with how our givers perceive our products. Monitoring and occasionally replying to customer questions is one of my ways to rectify this glaring shortcoming.

Last week, a question caught my eye.

Someone had written in asking for a "Give Card" supposedly sent by her husband (electronically) but never arrived.

My first reaction upon reading the query was: how in the world did her husband find the [Give Card](https://give.asia/give-card) feature to send in the first place?

Give Card is one of those experiments that we have come up with but did not find much traction. The idea is that for certain occasions such as a birthday or Lunar New Year, some people might not want more for themselves but rather to gift other people in need.

<img style="width: 100%" src="/assets/images/the-one-with-the-give-card/give-card_landing.jpeg">
> I know, we have not changed the Give Card’s design since February

For example, my wife can buy me an $88 Give Card. I can then choose to dedicate that $88 to someone in need on Give.Asia. We envisioned Give Card as a way to spread kindness, one pair of people at a time.

The idea did not get as much traction as I thought. Looking at how this feature has performed, there are probably two reasons why it didn’t work out:

1. Perhaps most people are not as receptive to the idea of giving forward as we thought.

2. From a technical standpoint, it might also look quite like phishing when receiving an email with a link to “claim” a gift.

We have never been sure if those reasons are the cause of Give Card’s low adoption rate. We decided to relegate the feature to the footer of our website.

Thus, it really surprised me that someone has even found where to buy one of these Give Cards. Apparently, these extremely motivated users bought a Give Card in 2021. This month, to celebrate their wedding anniversary, the wife, let’s call her Eve, sent a Give Card to her husband, who in turn wanted to send another Give Card for her. So it looks like the Give Card was used as designed, right? Apparently not, the second Give Card was never sent. So what happened?

* First try: looking at our records, it looks like the Give Card has not been paid for, hence it’s never sent. Confident as I am in our products, I told Eve that it looked like her husband’s payment did not go through. Eve then got back to me with the confirmation email that Give.Asia had sent, together with the bank’s proof of the deduction. So definitely I was wrong and at this point, at quite a big risk of instead of fixing the users’ issue, suggesting that it was their omission.

* Second try: quite out of my wits, I asked our engineers to help me look at it. They got back with the suggestion that the couple might have mixed up the payments. The payment confirmation was actually from the first (successful) purchase by Eve, instead of the (failed) payment by her husband. At this point, if you can still follow my story, it’s already quite confusing. However, learning from my mistake on the first try, I didn’t want to get back to our users and insisted again that it was not our fault.

* Third try: we looked at the records again, and this time we found the issue. Sometime last year, we added another payment processor to make it more convenient for our givers. We also made Apple Pay available on the new payment processor. Now, since Give Card has been neglected, it still works only with the old payment processor. Nonetheless, we still show Apple Pay as a payment option for Give Card. Eve’s husband paid for his Give Card with Apple Pay successfully. However, because it went through the new payment processor that the Give Card is not correctly connected with, our system still shows the Give Card as unpaid.

I was quite happy that we found the right cause of the bug. We even created a fix to not show Apple Pay for Give Card anymore. However, the question remained of how to help our customers with their missing Give Card. Here’s what I proposed to Eve: I would like to send her another Give Card to thank her for spotting the issue. She can give that card forward to a campaign of her wish. I would then assign her husband’s payment for his missing Give Card to the same campaign. I also explained that although this solution would have the same technical outcome, her husband’s donation is dedicated to a campaign under Eve’s name, so she would not have the same emotional gift of receiving the Give Card from her husband. I apologized for this and asked if it would be a good enough solution for her.

Eve and her husband were kind enough to accept the solution. I went ahead to send her a Give Card and then assigned her husband’s donation to the same campaign that she chose.

Eve was also so kind to give me a “Great” rating despite my initial wrong diagnosis and almost a week of waiting for the issue to be resolved.

<img style="width: 100%" src="/assets/images/the-one-with-the-give-card/missing-give-card.jpeg">

What have I learned from this episode? Quite a few things:

* Products can speak for themselves. This is an advice I happened to receive quite recently from our product advisors. I think it speaks somewhat about the case of Give Card. Despite being almost abandoned by us, it still can find its way to users.

* The best technical fix usually does not involve any coding. Could I have done something different to have Eve’s husband’s Give Card sent to her? Yes, and we already knew how that can be done. But that might involve another day or so to get the fix ready. My call in the end was not to go in that direction. Given it has already passed their anniversary, having the card resent would not have the same meaning. Making Eve and her husband wait another one or two days seemed like a terrible use of time for both them and our team.

There’s one more lesson: customer support is hard. Our support team’s goal is to keep a ticket’s resolution time under 24 hours on average, and they have been consistent in sticking with that goal. I got the impression that I might have wrecked that average.

See you in the next support ticket though. ;)
